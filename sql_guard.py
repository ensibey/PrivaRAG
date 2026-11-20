"""
Read-only SQL safety validator for PrivaRAG's natural-language SQL agent.

Prevents prompt-injected LLMs from executing destructive DDL/DML statements
(DROP, DELETE, UPDATE, INSERT, ALTER, ATTACH, PRAGMA) or stacked queries.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Tuple

FORBIDDEN_KEYWORDS = re.compile(
    r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|CREATE|REPLACE|TRUNCATE|ATTACH|DETACH|PRAGMA|VACUUM|REINDEX)\b",
    re.IGNORECASE,
)

ALLOWED_PREFIXES = ("SELECT", "WITH", "EXPLAIN")


@dataclass(frozen=True)
class SQLValidationResult:
    is_safe: bool
    normalized_sql: str
    reason: str = ""


def strip_sql_comments(sql: str) -> str:
    without_block = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    without_line = re.sub(r"--[^\n]*", " ", without_block)
    return " ".join(without_line.split()).strip()


def validate_readonly_sql(sql: str) -> SQLValidationResult:
    """Validate that an LLM-generated SQL statement is strictly a single read-only query."""
    cleaned = strip_sql_comments(sql or "")
    if not cleaned:
        return SQLValidationResult(False, "", "Empty SQL query.")

    # Allow a single trailing semicolon, reject stacked statements
    body = cleaned[:-1].strip() if cleaned.endswith(";") else cleaned
    if ";" in body:
        return SQLValidationResult(False, cleaned, "Multiple stacked SQL statements are forbidden.")

    first_token = body.split(None, 1)[0].upper()
    if first_token not in ALLOWED_PREFIXES:
        return SQLValidationResult(
            False,
            cleaned,
            f"Only read-only SELECT/WITH queries are allowed (got '{first_token}').",
        )

    match = FORBIDDEN_KEYWORDS.search(body)
    if match:
        return SQLValidationResult(
            False,
            cleaned,
            f"Forbidden SQL keyword detected: '{match.group(1).upper()}'.",
        )

    return SQLValidationResult(True, body, "OK")


def execute_readonly_sql(db_path: str | Path, sql: str, max_rows: int = 100) -> Tuple[List[str], List[Tuple[Any, ...]]]:
    """Validate and execute a read-only query using SQLite's native read-only URI mode."""
    check = validate_readonly_sql(sql)
    if not check.is_safe:
        raise ValueError(f"Blocked unsafe SQL query: {check.reason}")

    resolved = Path(db_path).resolve()
    conn = sqlite3.connect(f"file:{resolved.as_posix()}?mode=ro", uri=True)
    try:
        cursor = conn.execute(check.normalized_sql)
        columns = [desc[0] for desc in (cursor.description or [])]
        rows = cursor.fetchmany(max_rows)
        return columns, rows
    finally:
        conn.close()
