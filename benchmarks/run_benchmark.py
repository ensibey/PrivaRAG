"""
Reproducible Retrieval & Semantic Cache Benchmark for PrivaRAG.

Compares Okapi BM25, Dense Cosine Search, and Hybrid RRF on Recall@1, Recall@3,
MRR, NDCG@3, and mean query latency (ms), plus VectorSemanticCache latency.

Usage:
    python benchmarks/run_benchmark.py
    python benchmarks/run_benchmark.py --json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from benchmarks.corpus import ENTERPRISE_DOCS, EVAL_QUERIES
from bm25_retriever import OkapiBM25Retriever
from hybrid_retriever import HybridRRFRetriever
from semantic_cache import VectorSemanticCache


def _evaluate_method(rank_fn: Callable[[str], List[str]]) -> Dict[str, Any]:
    r1 = 0
    r3 = 0
    mrr_sum = 0.0
    ndcg3_sum = 0.0
    latencies_ms: List[float] = []

    for query, expected_id in EVAL_QUERIES:
        t0 = time.perf_counter()
        ranked_ids = rank_fn(query)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        if ranked_ids and ranked_ids[0] == expected_id:
            r1 += 1
        if expected_id in ranked_ids[:3]:
            r3 += 1
            rank = ranked_ids[:3].index(expected_id) + 1
            ndcg3_sum += 1.0 / math.log2(rank + 1)
        if expected_id in ranked_ids:
            mrr_sum += 1.0 / (ranked_ids.index(expected_id) + 1)

    n = len(EVAL_QUERIES)
    return {
        "recall_at_1": round(r1 / n, 4),
        "recall_at_3": round(r3 / n, 4),
        "mrr": round(mrr_sum / n, 4),
        "ndcg_at_3": round(ndcg3_sum / n, 4),
        "mean_latency_ms": round(sum(latencies_ms) / n, 3),
    }


def run_benchmark() -> Dict[str, Any]:
    doc_ids = [d[0] for d in ENTERPRISE_DOCS]
    texts = [d[1] for d in ENTERPRISE_DOCS]
    metas = [d[2] for d in ENTERPRISE_DOCS]

    bm25 = OkapiBM25Retriever()
    bm25.add_documents(doc_ids, texts, metas)

    hybrid = HybridRRFRetriever(bm25_weight=0.45, dense_weight=0.55, rrf_k=60)
    hybrid.add_documents(doc_ids, texts, metas)

    bm25_res = _evaluate_method(lambda q: [h.doc_id for h in bm25.search(q, top_k=5)])
    dense_res = _evaluate_method(lambda q: [doc_id for doc_id, _ in hybrid.search_dense(q, top_k=5)])
    hybrid_res = _evaluate_method(lambda q: [h.doc_id for h in hybrid.search(q, top_k=5)])

    # Measure VectorSemanticCache hit latency on repeated queries
    cache = VectorSemanticCache(similarity_threshold=0.88)
    for query, doc_id in EVAL_QUERIES:
        cache.store(query, f"Answer grounded in {doc_id}", [{"source": doc_id}])

    hit_latencies: List[float] = []
    hits = 0
    for query, _ in EVAL_QUERIES:
        t0 = time.perf_counter()
        res = cache.lookup(query)
        hit_latencies.append((time.perf_counter() - t0) * 1000.0)
        if res and res.get("cache_hit"):
            hits += 1

    return {
        "corpus_size": len(ENTERPRISE_DOCS),
        "num_queries": len(EVAL_QUERIES),
        "bm25": bm25_res,
        "dense": dense_res,
        "hybrid_rrf": hybrid_res,
        "semantic_cache": {
            "hit_rate": round(hits / len(EVAL_QUERIES), 4),
            "mean_hit_latency_ms": round(sum(hit_latencies) / len(hit_latencies), 3),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PrivaRAG retrieval and cache benchmark.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = run_benchmark()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"PrivaRAG Benchmark ({report['corpus_size']} docs, {report['num_queries']} queries)")
        print("=" * 68)
        print(f"{'Method':<16} | {'R@1':>6} | {'R@3':>6} | {'MRR':>6} | {'NDCG@3':>7} | {'Latency':>9}")
        print("-" * 68)
        for name, key in [("Okapi BM25", "bm25"), ("Dense Cosine", "dense"), ("Hybrid RRF", "hybrid_rrf")]:
            m = report[key]
            print(
                f"{name:<16} | {m['recall_at_1']*100:5.1f}% | {m['recall_at_3']*100:5.1f}% | "
                f"{m['mrr']:6.3f} | {m['ndcg_at_3']:7.3f} | {m['mean_latency_ms']:6.3f} ms"
            )
        sc = report["semantic_cache"]
        print("-" * 68)
        print(f"Semantic Cache: {sc['hit_rate']*100:.1f}% hit rate, {sc['mean_hit_latency_ms']:.3f} ms mean lookup")
    return 0


if __name__ == "__main__":
    sys.exit(main())
