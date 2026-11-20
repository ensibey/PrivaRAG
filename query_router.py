"""
Deterministic query intent router for PrivaRAG.

Routes incoming English or Turkish questions to the most appropriate engine:
- SQL: relational aggregations, table joins, counts, group-by, database queries
- SPREADSHEET: Excel/CSV workbook sheets, pivot-like column analytics
- DOC_RAG: unstructured policy, contract, legal, and architecture documents
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import List


class RouteTarget(str, Enum):
    DOC_RAG = "DOC_RAG"
    SQL = "SQL"
    SPREADSHEET = "SPREADSHEET"


SQL_PATTERNS = re.compile(
    r"\b(sql|sqlite|table|tablo|group by|join|select|count|sum|average|avg|toplam|ortalama|kaç tane|en yüksek|en düşük|revenue by|sales by|top \d+)\b",
    re.IGNORECASE,
)

SPREADSHEET_PATTERNS = re.compile(
    r"\b(excel|xlsx|csv|sheet|sayfa|sütun|column|row|satır|spreadsheet|pivot|dataframe|bütçe tablosu)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RouteDecision:
    target: RouteTarget
    confidence: float
    matched_signals: List[str]


def route_query(question: str, has_sql_db: bool = True, has_spreadsheet: bool = True) -> RouteDecision:
    """Classify a user question into DOC_RAG, SQL, or SPREADSHEET."""
    text = (question or "").strip()
    if not text:
        return RouteDecision(RouteTarget.DOC_RAG, 0.5, [])

    sheet_hits = [m.group(0).lower() for m in SPREADSHEET_PATTERNS.finditer(text)]
    sql_hits = [m.group(0).lower() for m in SQL_PATTERNS.finditer(text)]

    if has_spreadsheet and sheet_hits and len(sheet_hits) >= len(sql_hits):
        conf = min(0.98, 0.68 + 0.10 * len(sheet_hits))
        return RouteDecision(RouteTarget.SPREADSHEET, round(conf, 2), sheet_hits)

    if has_sql_db and sql_hits:
        conf = min(0.98, 0.65 + 0.10 * len(sql_hits))
        return RouteDecision(RouteTarget.SQL, round(conf, 2), sql_hits)

    return RouteDecision(RouteTarget.DOC_RAG, 0.85, [])
