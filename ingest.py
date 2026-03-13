"""
ingest.py — Document ingestion pipeline (v2).

PDF files: chunked per-page so each chunk carries an accurate page number.
Excel/CSV: each sheet/file converted to text, stored with sheet metadata.
All chunks embedded with nomic-embed-text via Ollama and saved to ChromaDB.
"""

import os
import sys
import tempfile
from pathlib import Path

import PyPDF2
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

from config import (
    EMBEDDING_MODEL,
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    OLLAMA_BASE_URL,
)

# ---------------------------------------------------------------------------
# File readers — return list of {"text", "page", "sheet"} dicts
# ---------------------------------------------------------------------------

def read_pdf(file_path: str) -> list[dict]:
    """Extract text page-by-page so each chunk knows its page number."""
    pages = []
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                pages.append({"text": text, "page": page_num + 1, "sheet": None})
    return pages


def read_excel(file_path: str) -> list[dict]:
    """Convert each sheet of an Excel file to a text block."""
    ext = Path(file_path).suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(file_path)
        df = df.dropna(how="all").dropna(axis=1, how="all")
        return [{"text": df.to_string(index=False), "page": None, "sheet": "CSV"}]

    sheets = []
    xls = pd.ExcelFile(file_path)
    for sheet_name in xls.sheet_names:
        df = xls.parse(sheet_name)
        df = df.dropna(how="all").dropna(axis=1, how="all")
        if df.empty:
            continue
        sheets.append({
            "text": df.to_string(index=False),
            "page": None,
            "sheet": sheet_name,
        })
    return sheets


def load_document(file_path: str) -> list[dict]:
    """Dispatch to the correct reader based on file extension."""
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return read_pdf(file_path)
    elif ext in (".xlsx", ".xls", ".csv"):
        return read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_sections(sections: list[dict], source_name: str) -> list[dict]:
    """
    Split each section (page/sheet) into overlapping chunks.
    Preserves page and sheet metadata from the source section.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = []
    for section in sections:
        sub_chunks = splitter.split_text(section["text"])
        for sub in sub_chunks:
            chunks.append({
                "text": sub,
                "source": source_name,
                "page": section.get("page"),        # int | None
                "sheet": section.get("sheet"),      # str | None
                "chunk_id": len(chunks),
            })
    return chunks


# ---------------------------------------------------------------------------
# Vector store helper
# ---------------------------------------------------------------------------

def get_vectorstore(embeddings: OllamaEmbeddings) -> Chroma:
    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ingest_file(file_path: str, progress_callback=None) -> int:
    """
    Full pipeline: read → chunk → embed → store in ChromaDB.

    Returns: number of chunks stored.
    """
    source_name = Path(file_path).name

    if progress_callback:
        progress_callback(0.10, f"Reading {source_name}...")

    sections = load_document(file_path)
    if not sections:
        raise ValueError(f"No text could be extracted from {source_name}.")

    if progress_callback:
        progress_callback(0.35, f"Splitting into chunks (found {len(sections)} sections)...")

    chunks = chunk_sections(sections, source_name)

    if progress_callback:
        progress_callback(0.55, f"Embedding {len(chunks)} chunks with {EMBEDDING_MODEL}...")

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)

    texts = [c["text"] for c in chunks]
    metadatas = []
    for c in chunks:
        meta = {"source": c["source"], "chunk_id": c["chunk_id"]}
        if c["page"] is not None:
            meta["page"] = c["page"]
        if c["sheet"] is not None:
            meta["sheet"] = c["sheet"]
        metadatas.append(meta)

    if progress_callback:
        progress_callback(0.80, "Writing to ChromaDB...")

    vs = get_vectorstore(embeddings)
    vs.add_texts(texts=texts, metadatas=metadatas)

    if progress_callback:
        progress_callback(1.0, f"Done! {len(chunks)} chunks from {source_name}.")

    return len(chunks)


def get_doc_count() -> int:
    """Total chunks currently in ChromaDB."""
    try:
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
        vs = get_vectorstore(embeddings)
        return vs._collection.count()
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <file_path>")
        sys.exit(1)

    def cli_cb(frac, msg):
        bar = "#" * int(frac * 30)
        print(f"  [{bar:<30}] {int(frac * 100):3d}%  {msg}")

    n = ingest_file(sys.argv[1], progress_callback=cli_cb)
    print(f"\n✅ {n} chunks stored.")
