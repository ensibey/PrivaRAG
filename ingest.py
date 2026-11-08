"""Document ingestion pipeline for PDF, Excel, and CSV sources into ChromaDB."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable, List, Optional

import fitz  # PyMuPDF
import pandas as pd
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

from config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
)

logger = logging.getLogger(__name__)


def load_pdf(file_path: str) -> List[Document]:
    """Extract page-level text from a PDF file via PyMuPDF."""
    docs: List[Document] = []
    pdf = fitz.open(file_path)
    filename = Path(file_path).name

    for page_num in range(len(pdf)):
        page = pdf[page_num]
        text = page.get_text("text").strip()
        if text:
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": filename, "page": page_num + 1},
                )
            )
    pdf.close()
    return docs


def load_excel(file_path: str) -> List[Document]:
    """Convert non-empty Excel sheets or CSV files into structured text documents."""
    docs: List[Document] = []
    filename = Path(file_path).name
    ext = Path(file_path).suffix.lower()

    if ext == ".csv":
        df = pd.read_csv(file_path).dropna(how="all")
        if not df.empty:
            headers = " | ".join(str(col) for col in df.columns)
            rows = [
                " | ".join(str(val) for val in row)
                for row in df.itertuples(index=False)
            ]
            content = f"Source: {filename}\nColumns: {headers}\n\n" + "\n".join(rows)
            docs.append(
                Document(
                    page_content=content,
                    metadata={"source": filename, "sheet": "CSV"},
                )
            )
        return docs

    xls = pd.ExcelFile(file_path)
    for sheet_name in xls.sheet_names:
        df = xls.parse(sheet_name).dropna(how="all")
        if df.empty:
            continue
        headers = " | ".join(str(col) for col in df.columns)
        rows = [
            " | ".join(str(val) for val in row)
            for row in df.itertuples(index=False)
        ]
        content = f"Sheet: {sheet_name}\nColumns: {headers}\n\n" + "\n".join(rows)
        docs.append(
            Document(
                page_content=content,
                metadata={"source": filename, "sheet": sheet_name},
            )
        )

    return docs


def split_documents(docs: List[Document]) -> List[Document]:
    """Split documents into overlapping chunks while preserving source metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx
    return chunks


def get_vectorstore(embeddings: OllamaEmbeddings) -> Chroma:
    """Return the persistent ChromaDB collection handle."""
    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )


def ingest_file(
    file_path: str,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> int:
    """Load, chunk, embed, and persist a document into ChromaDB."""

    def _report(pct: float, msg: str) -> None:
        if progress_callback:
            progress_callback(pct, msg)

    ext = Path(file_path).suffix.lower()
    _report(0.10, f"Reading {Path(file_path).name}...")

    if ext == ".pdf":
        raw_docs = load_pdf(file_path)
    elif ext in (".xlsx", ".xls", ".csv"):
        raw_docs = load_excel(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    if not raw_docs:
        _report(1.0, "No text content extracted.")
        return 0

    _report(0.35, f"Splitting {len(raw_docs)} section(s) into chunks...")
    chunks = split_documents(raw_docs)

    _report(0.55, f"Computing embeddings for {len(chunks)} chunks via {EMBEDDING_MODEL}...")
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )

    _report(0.75, "Persisting vectors to ChromaDB...")
    vectorstore = get_vectorstore(embeddings)
    vectorstore.add_documents(chunks)

    _report(1.0, f"Indexed {len(chunks)} chunks from {Path(file_path).name}.")
    return len(chunks)


def get_doc_count() -> int:
    """Return the total number of indexed chunks in ChromaDB."""
    if not os.path.exists(CHROMA_PERSIST_DIR):
        return 0
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
    vs = get_vectorstore(embeddings)
    return vs._collection.count()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ingest.py <path_to_pdf_or_excel>")
        sys.exit(1)

    target = sys.argv[1]
    count = ingest_file(
        target,
        progress_callback=lambda p, m: print(f"[{int(p * 100):3d}%] {m}"),
    )
    print(f"Total chunks indexed: {count}")
