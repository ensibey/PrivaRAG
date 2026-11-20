"""Regression tests verifying PrivaRAG hybrid retrieval and semantic cache benchmark quality."""

from benchmarks.run_benchmark import run_benchmark


def test_hybrid_rrf_benchmark_meets_high_recall_and_mrr():
    report = run_benchmark()
    hybrid = report["hybrid_rrf"]
    assert hybrid["recall_at_1"] >= 0.90
    assert hybrid["recall_at_3"] == 1.0
    assert hybrid["mrr"] >= 0.95
    assert report["semantic_cache"]["hit_rate"] == 1.0
    assert report["semantic_cache"]["mean_hit_latency_ms"] < 15.0
