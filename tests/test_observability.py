"""Tests for PrivaRAG structured tracing, tenant token-bucket limiter, and SSE token streaming."""

import json
from pathlib import Path

from observability import RAGTracer, TenantRateLimiter, stream_sse_tokens


def test_rag_tracer_and_jsonl_export(tmp_path: Path):
    log_file = tmp_path / "rag_traces.jsonl"
    tracer = RAGTracer(trace_id="rag-01", log_path=log_file)

    with tracer.stage("pii_redaction") as m:
        m["redacted"] = 2
    with tracer.stage("hybrid_rrf") as m:
        m["hits"] = 3

    rep = tracer.report()
    assert rep["trace_id"] == "rag-01"
    assert len(rep["stages"]) == 2
    lines = [json.loads(l) for l in log_file.read_text(encoding="utf-8").splitlines()]
    assert [l["stage"] for l in lines] == ["pii_redaction", "hybrid_rrf"]


def test_tenant_rate_limiter_and_sse_streaming():
    limiter = TenantRateLimiter(capacity=2, refill_per_sec=1.0)
    assert limiter.allow("team-a", now=0.0)[0] is True
    assert limiter.allow("team-a", now=0.0)[0] is True
    assert limiter.allow("team-a", now=0.0)[0] is False
    assert limiter.allow("team-a", now=1.2)[0] is True

    frames = list(stream_sse_tokens("Revenue reached 12.4M EUR", trace_id="t1"))
    assert len(frames) == 5  # 4 tokens + 1 done event
    assert frames[-1].startswith("event: done")
