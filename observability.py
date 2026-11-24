"""
Observability, Token-Bucket Rate Limiting, and Streaming Answer Generator for PrivaRAG.

Zero external dependencies so air-gapped deployments and CI suites can trace
per-stage RAG latency (PII -> Cache -> BM25/RRF -> Faithfulness), enforce
per-tenant request quotas, and stream grounded responses token-by-token.
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple


@dataclass
class RAGSpan:
    trace_id: str
    stage: str
    duration_ms: float
    attributes: Dict[str, Any] = field(default_factory=dict)


class RAGTracer:
    """Records structured per-stage RAG latency and metadata with optional JSONL export."""

    def __init__(self, trace_id: Optional[str] = None, log_path: Optional[Path] = None) -> None:
        self.trace_id = trace_id or uuid.uuid4().hex[:16]
        self.log_path = log_path
        self.spans: List[RAGSpan] = []

    @contextmanager
    def stage(self, name: str, **attrs: Any) -> Iterator[Dict[str, Any]]:
        t0 = time.perf_counter()
        meta = dict(attrs)
        try:
            yield meta
        finally:
            ms = round((time.perf_counter() - t0) * 1000.0, 3)
            span = RAGSpan(trace_id=self.trace_id, stage=name, duration_ms=ms, attributes=meta)
            self.spans.append(span)
            if self.log_path is not None:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.log_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(asdict(span)) + "\n")

    def report(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "total_ms": round(sum(s.duration_ms for s in self.spans), 3),
            "stages": [asdict(s) for s in self.spans],
        }


class TenantRateLimiter:
    """Per-tenant Token-Bucket rate limiter for multi-user enterprise PrivaRAG endpoints."""

    def __init__(self, capacity: int = 20, refill_per_sec: float = 0.5) -> None:
        self.capacity = float(capacity)
        self.refill_per_sec = float(refill_per_sec)
        self._state: Dict[str, Tuple[float, float]] = {}

    def allow(self, tenant_id: str, now: Optional[float] = None) -> Tuple[bool, int]:
        ts = time.monotonic() if now is None else now
        tokens, last_ts = self._state.get(tenant_id, (self.capacity, ts))
        tokens = min(self.capacity, tokens + max(0.0, ts - last_ts) * self.refill_per_sec)
        if tokens >= 1.0:
            tokens -= 1.0
            self._state[tenant_id] = (tokens, ts)
            return True, int(tokens)
        self._state[tenant_id] = (tokens, ts)
        return False, 0


def stream_sse_tokens(text: str, trace_id: str) -> Iterator[str]:
    """Yield Server-Sent Events (SSE) token chunks for non-blocking streaming UIs."""
    for word in (text or "").split():
        yield f"event: token\ndata: {json.dumps({'token': word + ' '})}\n\n"
    yield f"event: done\ndata: {json.dumps({'trace_id': trace_id})}\n\n"
