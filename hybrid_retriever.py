"""
Hybrid Sparse + Dense Retriever using Reciprocal Rank Fusion (RRF).

Combines Okapi BM25 lexical ranking with cosine-similarity dense vector search.
Works fully offline with deterministic character n-gram embeddings by default,
or plugs into any Ollama/SentenceTransformers embedding function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

from bm25_retriever import OkapiBM25Retriever
from semantic_cache import _cosine_similarity, deterministic_ngram_embedding


@dataclass
class HybridHit:
    doc_id: str
    rrf_score: float
    bm25_rank: Optional[int]
    dense_rank: Optional[int]
    dense_similarity: float
    text: str
    metadata: Dict[str, object]


class HybridRRFRetriever:
    """Combines sparse Okapi BM25 and dense cosine retrieval via Reciprocal Rank Fusion."""

    def __init__(
        self,
        bm25_weight: float = 0.45,
        dense_weight: float = 0.55,
        rrf_k: int = 60,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
    ) -> None:
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.rrf_k = rrf_k
        self.embed_fn = embed_fn or deterministic_ngram_embedding

        self._bm25 = OkapiBM25Retriever()
        self._doc_ids: List[str] = []
        self._texts: List[str] = []
        self._metadata: List[Dict[str, object]] = []
        self._vectors: List[List[float]] = []

    def add_documents(
        self,
        doc_ids: Sequence[str],
        texts: Sequence[str],
        metadatas: Optional[Sequence[Dict[str, object]]] = None,
    ) -> None:
        metas = list(metadatas) if metadatas is not None else [{} for _ in texts]
        self._bm25.add_documents(doc_ids, texts, metas)
        for doc_id, text, meta in zip(doc_ids, texts, metas):
            self._doc_ids.append(doc_id)
            self._texts.append(text)
            self._metadata.append(dict(meta))
            self._vectors.append(self.embed_fn(text))

    def search_dense(self, query: str, top_k: int = 10) -> List[tuple[str, float]]:
        if not self._doc_ids or top_k <= 0:
            return []
        q_vec = self.embed_fn(query)
        scored = [
            (doc_id, _cosine_similarity(q_vec, vec))
            for doc_id, vec in zip(self._doc_ids, self._vectors)
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]

    def search(self, query: str, top_k: int = 5) -> List[HybridHit]:
        if not self._doc_ids or top_k <= 0:
            return []

        pool_k = max(top_k * 3, 15)
        bm25_hits = self._bm25.search(query, top_k=pool_k)
        dense_hits = self.search_dense(query, top_k=pool_k)

        bm25_ranks = {hit.doc_id: rank for rank, hit in enumerate(bm25_hits, start=1)}
        dense_ranks = {doc_id: rank for rank, (doc_id, _) in enumerate(dense_hits, start=1)}
        dense_sims = {doc_id: sim for doc_id, sim in dense_hits}

        doc_lookup = {
            doc_id: (text, meta)
            for doc_id, text, meta in zip(self._doc_ids, self._texts, self._metadata)
        }

        candidate_ids = set(bm25_ranks.keys()) | set(dense_ranks.keys())
        fused: List[HybridHit] = []

        for doc_id in candidate_ids:
            score = 0.0
            b_rank = bm25_ranks.get(doc_id)
            d_rank = dense_ranks.get(doc_id)
            if b_rank is not None:
                score += self.bm25_weight / (self.rrf_k + b_rank)
            if d_rank is not None:
                score += self.dense_weight / (self.rrf_k + d_rank)

            text, meta = doc_lookup[doc_id]
            fused.append(
                HybridHit(
                    doc_id=doc_id,
                    rrf_score=round(score, 6),
                    bm25_rank=b_rank,
                    dense_rank=d_rank,
                    dense_similarity=round(dense_sims.get(doc_id, 0.0), 4),
                    text=text,
                    metadata=meta,
                )
            )

        fused.sort(key=lambda h: h.rrf_score, reverse=True)
        return fused[:top_k]
