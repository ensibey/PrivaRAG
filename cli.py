"""
Standalone offline CLI for PrivaRAG local search, PII redaction, SQL safety auditing, and query routing.

Usage:
    python cli.py search docs/ "air-gapped deployment"
    python cli.py redact "Email: alice@corp.com"
    python cli.py audit-sql "SELECT * FROM sales; DROP TABLE sales;"
    python cli.py route "Show total revenue by region in the sales table"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from chunker import RecursiveChunker
from hybrid_retriever import HybridRRFRetriever
from pii_guard import redact_pii
from query_router import route_query
from sql_guard import validate_readonly_sql


def cmd_search(target: Path, query: str, top_k: int, as_json: bool) -> int:
    if not target.exists():
        print(f"error: path not found: {target}", file=sys.stderr)
        return 2

    files = [target] if target.is_file() else sorted(
        p for p in target.rglob("*") if p.is_file() and p.suffix.lower() in {".md", ".txt", ".py", ".csv"}
    )
    chunker = RecursiveChunker(chunk_size=500, chunk_overlap=80)
    retriever = HybridRRFRetriever()

    doc_ids: List[str] = []
    texts: List[str] = []
    metas: List[dict] = []
    for f in files:
        content = f.read_text(encoding="utf-8", errors="ignore")
        for chunk in chunker.chunk_text(content, source=f.as_posix()):
            doc_ids.append(chunk.chunk_id)
            texts.append(chunk.text)
            metas.append({"source": chunk.source, "chunk_index": chunk.chunk_index})

    retriever.add_documents(doc_ids, texts, metas)
    hits = retriever.search(query, top_k=top_k)

    payload = [
        {
            "doc_id": h.doc_id,
            "source": h.metadata.get("source"),
            "rrf_score": h.rrf_score,
            "bm25_rank": h.bm25_rank,
            "dense_rank": h.dense_rank,
            "preview": h.text[:160],
        }
        for h in hits
    ]

    if as_json:
        print(json.dumps({"query": query, "files_indexed": len(files), "hits": payload}, indent=2))
    else:
        for idx, h in enumerate(payload, start=1):
            print(f"[{idx}] ({h['rrf_score']:.5f}) {h['source']} — {h['preview']}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="privarag", description="Offline PrivaRAG retrieval & privacy toolkit.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_search = sub.add_parser("search", help="Run offline hybrid BM25 + dense RRF search over local files.")
    p_search.add_argument("target", help="File or directory of .md/.txt/.csv files")
    p_search.add_argument("query", help="Natural language search query")
    p_search.add_argument("-k", "--top-k", type=int, default=5)
    p_search.add_argument("--json", action="store_true")

    p_redact = sub.add_parser("redact", help="Mask PII (emails, phones, IBAN, TCKN) in text.")
    p_redact.add_argument("text")

    p_sql = sub.add_parser("audit-sql", help="Validate whether a SQL statement is strictly read-only.")
    p_sql.add_argument("sql")

    p_route = sub.add_parser("route", help="Classify a question into DOC_RAG, SQL, or SPREADSHEET.")
    p_route.add_argument("question")

    args = parser.parse_args(argv)

    if args.command == "search":
        return cmd_search(Path(args.target), args.query, args.top_k, args.json)
    if args.command == "redact":
        report = redact_pii(args.text)
        print(json.dumps({"redacted_text": report.redacted_text, "counts": report.counts}, indent=2))
        return 0
    if args.command == "audit-sql":
        res = validate_readonly_sql(args.sql)
        print(json.dumps({"is_safe": res.is_safe, "reason": res.reason, "sql": res.normalized_sql}, indent=2))
        return 0 if res.is_safe else 1
    if args.command == "route":
        dec = route_query(args.question)
        print(json.dumps({"target": dec.target.value, "confidence": dec.confidence, "signals": dec.matched_signals}, indent=2))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
