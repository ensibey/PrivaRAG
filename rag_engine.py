"""
rag_engine.py — Conversational RAG chain (v3).

Changes in v3:
  • Hybrid retrieval: BM25 (keyword) + ChromaDB (semantic) via EnsembleRetriever
    — weights: 40% BM25 / 60% vector — catches exact phrase matches like "Invoice D-102"
  • Conversation summary memory: when chat history exceeds MAX_HISTORY_TURNS,
    all older turns are summarised into 2 sentences by the local LLM before being
    sent as context — prevents context-window overflow on long sessions.
"""

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

from config import (
    LLM_MODEL,
    EMBEDDING_MODEL,
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    TOP_K_RESULTS,
    OLLAMA_BASE_URL,
    MAX_HISTORY_TURNS,
)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(
    """Given the conversation history and the new follow-up question,
rephrase the follow-up question as a complete standalone question.

Chat History:
{chat_history}

Follow Up Question: {question}
Standalone Question:"""
)

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a precise assistant. Answer ONLY based on the provided
document context. If the answer is not in the context, say:
"I could not find this information in the uploaded documents."

Document Context:
{context}

Question: {question}

Answer:""",
)


# ---------------------------------------------------------------------------
# History compression (conversation summary memory)
# ---------------------------------------------------------------------------

def compress_history(
    history: list[tuple[str, str]],
    llm: ChatOllama,
) -> list[tuple[str, str]]:
    """
    If history exceeds MAX_HISTORY_TURNS, summarise the older turns into
    two sentences using the local LLM, then prepend as a synthetic turn.
    This keeps context-window usage bounded for long sessions.
    """
    if len(history) <= MAX_HISTORY_TURNS:
        return history

    old_turns = history[:-MAX_HISTORY_TURNS]
    recent_turns = history[-MAX_HISTORY_TURNS:]

    old_text = "\n".join(
        f"User: {h}\nAI: {a}" for h, a in old_turns
    )
    summary_prompt = (
        "Summarise the following conversation in exactly 2 sentences, "
        "preserving all key facts, names, and figures:\n\n" + old_text
    )
    summary = llm.invoke(summary_prompt).content.strip()
    return [("(Earlier conversation summary)", summary)] + recent_turns


# ---------------------------------------------------------------------------
# Hybrid retriever builder
# ---------------------------------------------------------------------------

def _build_hybrid_retriever(vectorstore: Chroma) -> EnsembleRetriever:
    """
    Combine BM25 (keyword/exact) and ChromaDB (semantic/dense) retrievers.

    BM25 excels at exact matches ("Invoice D-102", product codes, names).
    Vector search excels at semantic similarity ("what are the revenue figures?").
    EnsembleRetriever merges and re-ranks results from both.
    """
    collection = vectorstore.get()
    docs_texts = collection.get("documents", [])
    docs_metas = collection.get("metadatas", [])

    if not docs_texts:
        # Fallback to vector-only when the DB is empty
        return vectorstore.as_retriever(search_kwargs={"k": TOP_K_RESULTS})

    bm25 = BM25Retriever.from_texts(docs_texts, metadatas=docs_metas)
    bm25.k = TOP_K_RESULTS

    vector_ret = vectorstore.as_retriever(search_kwargs={"k": TOP_K_RESULTS})

    return EnsembleRetriever(
        retrievers=[bm25, vector_ret],
        weights=[0.4, 0.6],   # 40% keyword boost + 60% semantic
    )


# ---------------------------------------------------------------------------
# Chain builder
# ---------------------------------------------------------------------------

def build_chain(llm: ChatOllama, embeddings: OllamaEmbeddings) -> ConversationalRetrievalChain:
    vectorstore = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    retriever = _build_hybrid_retriever(vectorstore)

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True,
        verbose=False,
    )


# ---------------------------------------------------------------------------
# Public query function
# ---------------------------------------------------------------------------

def query(question: str, chat_history: list[tuple[str, str]] | None = None) -> dict:
    """
    Run a hybrid RAG query with conversation summary memory.

    Args:
        question:     Current user question.
        chat_history: List of (human_str, ai_str) tuples from prior turns.

    Returns:
        {"answer": str, "sources": list[dict], "history_compressed": bool}
    """
    llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.1)
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)

    raw_history = chat_history or []
    compressed = compress_history(raw_history, llm)
    history_was_compressed = len(compressed) < len(raw_history)

    chain = build_chain(llm, embeddings)
    result = chain.invoke({
        "question": question,
        "chat_history": compressed,
    })

    answer = result.get("answer", "No answer generated.")
    raw_docs = result.get("source_documents", [])

    sources = []
    seen: set[tuple] = set()
    for doc in raw_docs:
        text = doc.page_content.strip()
        key = (doc.metadata.get("source", ""), text[:80])
        if key in seen:
            continue
        seen.add(key)
        sources.append({
            "source":   doc.metadata.get("source", "Unknown"),
            "page":     doc.metadata.get("page"),     # int | None
            "sheet":    doc.metadata.get("sheet"),    # str | None
            "chunk_id": doc.metadata.get("chunk_id", 0),
            "text":     text,
        })

    return {
        "answer": answer,
        "sources": sources,
        "history_compressed": history_was_compressed,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    history: list[tuple[str, str]] = []
    print("Local RAG Chat (Hybrid Search + Summary Memory) — type 'quit' to exit\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in ("quit", "exit"):
            break
        out = query(q, history)
        print(f"\nAI: {out['answer']}\n")
        if out["history_compressed"]:
            print("  ℹ️  (Older conversation was summarised to save context)")
        for s in out["sources"]:
            loc = f"Page {s['page']}" if s["page"] else s.get("sheet", "")
            print(f"  📄 {s['source']} {loc}")
        print()
        history.append((q, out["answer"]))
