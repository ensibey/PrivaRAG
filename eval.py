"""
eval.py — Local RAG Evaluation Script

Measures 3 key metrics using only local models (no cloud APIs):

  1. Faithfulness Score
     "Does every claim in the answer appear in the retrieved context?"
     Method: LLM-as-judge — for each factual sentence in the answer,
     ask the local LLM whether it is directly supported by the context.
     Score = supported_sentences / total_sentences  (0.0 – 1.0)

  2. Answer Relevancy Score
     "Does the answer actually address the question asked?"
     Method: Cosine similarity between the question embedding and the answer
     embedding. High similarity = the answer stays on-topic.
     Score = cosine_similarity(embed(question), embed(answer))  (0.0 – 1.0)

  3. Context Recall
     "Did the retriever pull in chunks that contain the answer?"
     Method: For each retrieved context chunk, ask the LLM whether the chunk
     contributed to the answer. Score = useful_chunks / total_chunks

Usage:
    python eval.py                   # runs built-in sample test set
    python eval.py --file tests.json # runs a custom JSON test file

Test file format (JSON array):
    [
      {
        "question": "What was the total revenue in Q3?",
        "answer": "The total revenue in Q3 was $4.2M.",      (optional gold answer)
        "contexts": ["Revenue Report Q3: total $4.2M ..."]   (optional override)
      }
    ]
"""

import json
import sys
import time
import argparse
from pathlib import Path

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import numpy as np

from config import (
    LLM_MODEL, EMBEDDING_MODEL, OLLAMA_BASE_URL,
    CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME, TOP_K_RESULTS,
)

# ---------------------------------------------------------------------------
# Models (initialised once)
# ---------------------------------------------------------------------------

def _get_llm():
    return ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)

def _get_embedder():
    return OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)

# ---------------------------------------------------------------------------
# Retrieval helper
# ---------------------------------------------------------------------------

def retrieve_contexts(question: str, k: int = TOP_K_RESULTS) -> list[str]:
    """Pull the top-K context chunks from ChromaDB for a question."""
    embedder = _get_embedder()
    vs = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embedder,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    docs = vs.similarity_search(question, k=k)
    return [d.page_content for d in docs]

# ---------------------------------------------------------------------------
# Metric 1: Faithfulness
# ---------------------------------------------------------------------------

FAITHFULNESS_PROMPT = """\
Context (retrieved from documents):
{context}

Answer being evaluated:
{answer}

Task: For each factual claim in the Answer, determine whether it is DIRECTLY
supported by the Context. Reply with a JSON object in this exact format:
{{"supported": <number of supported claims>, "total": <total factual claims>}}

Reply ONLY with the JSON object, no explanation."""

def faithfulness_score(answer: str, contexts: list[str], llm) -> float:
    """Score 0-1: what fraction of answer claims are grounded in the context."""
    if not answer.strip() or not contexts:
        return 0.0
    context_text = "\n---\n".join(contexts)
    prompt = FAITHFULNESS_PROMPT.format(context=context_text[:3000], answer=answer)
    try:
        response = llm.invoke(prompt).content.strip()
        # Extract JSON robustly
        start = response.find("{")
        end = response.rfind("}") + 1
        data = json.loads(response[start:end])
        total = max(data.get("total", 1), 1)
        return round(data.get("supported", 0) / total, 3)
    except Exception:
        return -1.0  # parse error


# ---------------------------------------------------------------------------
# Metric 2: Answer Relevancy
# ---------------------------------------------------------------------------

def cosine_similarity(a: list[float], b: list[float]) -> float:
    a_arr = np.array(a)
    b_arr = np.array(b)
    denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if denom == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / denom)


def answer_relevancy_score(question: str, answer: str, embedder) -> float:
    """Score 0-1: semantic similarity between question and answer."""
    if not question.strip() or not answer.strip():
        return 0.0
    try:
        q_emb = embedder.embed_query(question)
        a_emb = embedder.embed_query(answer)
        return round(cosine_similarity(q_emb, a_emb), 3)
    except Exception:
        return -1.0


# ---------------------------------------------------------------------------
# Metric 3: Context Recall
# ---------------------------------------------------------------------------

CONTEXT_RECALL_PROMPT = """\
Question: {question}
Answer: {answer}
Context chunk: {chunk}

Did this context chunk meaningfully contribute information used in the answer?
Reply with exactly one word: YES or NO."""

def context_recall_score(question: str, answer: str, contexts: list[str], llm) -> float:
    """Score 0-1: fraction of retrieved chunks that contributed to the answer."""
    if not contexts:
        return 0.0
    useful = 0
    for chunk in contexts:
        prompt = CONTEXT_RECALL_PROMPT.format(
            question=question, answer=answer, chunk=chunk[:600]
        )
        try:
            resp = llm.invoke(prompt).content.strip().upper()
            if resp.startswith("YES"):
                useful += 1
        except Exception:
            pass
    return round(useful / len(contexts), 3)


# ---------------------------------------------------------------------------
# Full evaluation runner
# ---------------------------------------------------------------------------

def evaluate_one(item: dict, llm, embedder) -> dict:
    """Evaluate a single question/answer pair."""
    question = item["question"]
    answer = item.get("answer") or ""

    # If no answer provided, generate one via RAG
    if not answer:
        from rag_engine import query as rag_query
        result = rag_query(question)
        answer = result["answer"]
        contexts = [s["text"] for s in result["sources"]]
    else:
        contexts = item.get("contexts") or retrieve_contexts(question)

    f_score = faithfulness_score(answer, contexts, llm)
    r_score = answer_relevancy_score(question, answer, embedder)
    c_score = context_recall_score(question, answer, contexts, llm)

    return {
        "question": question,
        "answer": answer[:200] + ("..." if len(answer) > 200 else ""),
        "faithfulness":       f_score,
        "answer_relevancy":   r_score,
        "context_recall":     c_score,
        "avg_score":          round(
            sum(s for s in [f_score, r_score, c_score] if s >= 0) /
            max(sum(1 for s in [f_score, r_score, c_score] if s >= 0), 1), 3
        ),
    }


def run_evaluation(test_items: list[dict]) -> dict:
    """Run evaluation on a list of test items and print a report."""
    llm = _get_llm()
    embedder = _get_embedder()

    print("\n" + "=" * 65)
    print("  LOCAL RAG EVALUATION REPORT")
    print("=" * 65)
    print(f"  Model: {LLM_MODEL}   Embedder: {EMBEDDING_MODEL}")
    print(f"  Tests: {len(test_items)}   Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    results = []
    for i, item in enumerate(test_items, 1):
        print(f"\n[{i}/{len(test_items)}] Evaluating: {item['question'][:60]}…")
        t0 = time.time()
        row = evaluate_one(item, llm, embedder)
        elapsed = time.time() - t0
        results.append(row)

        def fmt(v):
            return f"{v:.1%}" if v >= 0 else "ERR"

        print(f"  Faithfulness:     {fmt(row['faithfulness'])}")
        print(f"  Answer Relevancy: {fmt(row['answer_relevancy'])}")
        print(f"  Context Recall:   {fmt(row['context_recall'])}")
        print(f"  Average:          {fmt(row['avg_score'])}   ({elapsed:.1f}s)")

    # Aggregate
    def agg(key):
        vals = [r[key] for r in results if r[key] >= 0]
        return round(sum(vals) / len(vals), 3) if vals else -1.0

    summary = {
        "model": LLM_MODEL,
        "num_tests": len(results),
        "avg_faithfulness":     agg("faithfulness"),
        "avg_answer_relevancy": agg("answer_relevancy"),
        "avg_context_recall":   agg("context_recall"),
        "overall_avg":          agg("avg_score"),
        "results": results,
    }

    print(f"\n{'=' * 65}")
    print("  SUMMARY")
    print(f"{'=' * 65}")
    print(f"  Faithfulness:     {summary['avg_faithfulness']:.1%}")
    print(f"  Answer Relevancy: {summary['avg_answer_relevancy']:.1%}")
    print(f"  Context Recall:   {summary['avg_context_recall']:.1%}")
    print(f"  ── Overall Score: {summary['overall_avg']:.1%}")
    print(f"{'=' * 65}\n")

    return summary


# ---------------------------------------------------------------------------
# Default test cases (used when no JSON file provided)
# ---------------------------------------------------------------------------

DEFAULT_TESTS = [
    {
        "question": "What is the main topic of the uploaded documents?",
    },
    {
        "question": "Summarise the key findings in two sentences.",
    },
    {
        "question": "Are there any numbers or statistics mentioned?",
    },
]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local RAG Evaluator")
    parser.add_argument("--file", "-f", type=str, default=None,
                        help="Path to a JSON test file")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Save results to a JSON file")
    args = parser.parse_args()

    if args.file:
        with open(args.file, encoding="utf-8") as fh:
            test_items = json.load(fh)
        print(f"Loaded {len(test_items)} tests from {args.file}")
    else:
        print("No test file provided — running built-in demo tests.")
        test_items = DEFAULT_TESTS

    summary = run_evaluation(test_items)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False)
        print(f"Results saved to {args.output}")
