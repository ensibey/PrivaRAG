"""
Zero-dependency Okapi BM25 sparse lexical retriever for PrivaRAG.

Supports both English and Turkish Unicode tokenization and exact IDF + term-frequency
saturation scoring (k1=1.5, b=0.75).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Sequence

TOKEN_PATTERN = re.compile(r"[0-9a-zA-ZçğıöşüÇĞİÖŞÜ]+")


def tokenize_lexical(text: str) -> List[str]:
    lowered = text.replace("İ", "i").replace("I", "ı").lower()
    return TOKEN_PATTERN.findall(lowered)


@dataclass
class BM25Hit:
    doc_id: str
    score: float
    text: str
    metadata: Dict[str, object]


class OkapiBM25Retriever:
    """In-memory Okapi BM25 index over document chunks."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._doc_ids: List[str] = []
        self._texts: List[str] = []
        self._metadata: List[Dict[str, object]] = []
        self._term_freqs: List[Counter] = []
        self._doc_lens: List[int] = []
        self._doc_freq: Counter = Counter()
        self._avgdl: float = 0.0

    def add_documents(
        self,
        doc_ids: Sequence[str],
        texts: Sequence[str],
        metadatas: Sequence[Dict[str, object]] | None = None,
    ) -> None:
        metas = list(metadatas) if metadatas is not None else [{} for _ in texts]
        for doc_id, text, meta in zip(doc_ids, texts, metas):
            tokens = tokenize_lexical(text)
            tf = Counter(tokens)
            self._doc_ids.append(doc_id)
            self._texts.append(text)
            self._metadata.append(dict(meta))
            self._term_freqs.append(tf)
            self._doc_lens.append(len(tokens))
            for term in tf.keys():
                self._doc_freq[term] += 1

        total_docs = len(self._doc_lens)
        self._avgdl = (sum(self._doc_lens) / total_docs) if total_docs > 0 else 0.0

    def _idf(self, term: str) -> float:
        n_docs = len(self._doc_lens)
        df = self._doc_freq.get(term, 0)
        if df == 0 or n_docs == 0:
            return 0.0
        return math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 5) -> List[BM25Hit]:
        if not self._doc_ids or top_k <= 0:
            return []
        q_tokens = tokenize_lexical(query)
        if not q_tokens:
            return []

        scored: List[BM25Hit] = []
        for idx, tf in enumerate(self._term_freqs):
            dl = self._doc_lens[idx]
            score = 0.0
            for term in q_tokens:
                freq = tf.get(term, 0)
                if freq == 0:
                    continue
                idf = self._idf(term)
                denom = freq + self.k1 * (1.0 - self.b + self.b * (dl / (self._avgdl or 1.0)))
                score += idf * (freq * (self.k1 + 1.0)) / denom

            if score > 0.0:
                scored.append(
                    BM25Hit(
                        doc_id=self._doc_ids[idx],
                        score=round(score, 6),
                        text=self._texts[idx],
                        metadata=self._metadata[idx],
                    )
                )

        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:top_k]
