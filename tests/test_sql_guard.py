"""Tests for read-only SQLite safety validator."""

import sqlite3
from pathlib import Path

import pytest

from sql_guard import execute_readonly_sql, validate_readonly_sql


def test_allows_valid_select_and_cte():
    assert validate_readonly_sql("SELECT region, SUM(revenue) FROM sales GROUP BY region;").is_safe
    assert validate_readonly_sql("WITH top AS (SELECT * FROM sales) SELECT * FROM top").is_safe


@pytest.mark.parametrize(
    "bad_sql",
    [
        "DROP TABLE users;",
        "DELETE FROM sales WHERE id = 1",
        "UPDATE users SET role = 'admin'",
        "INSERT INTO logs VALUES ('x')",
        "SELECT * FROM sales; DROP TABLE sales;",
        "PRAGMA table_info(users)",
        "ATTACH DATABASE '/tmp/pwn.db' AS pwn",
        "   ",
    ],
)
def test_blocks_destructive_and_stacked_sql(bad_sql: str):
    res = validate_readonly_sql(bad_sql)
    assert res.is_safe is False
    assert res.reason != ""


def test_execute_readonly_sql_against_real_sqlite(tmp_path: Path):
    db_file = tmp_path / "finance.db"
    conn = sqlite3.connect(db_file)
    conn.execute("CREATE TABLE sales (region TEXT, amount REAL)")
    conn.execute("INSERT INTO sales VALUES ('EMEA', 1200.5), ('APAC', 800.0)")
    conn.commit()
    conn.close()

    cols, rows = execute_readonly_sql(db_file, "SELECT region, amount FROM sales ORDER BY amount DESC")
    assert cols == ["region", "amount"]
    assert rows[0] == ("EMEA", 1200.5)

    with pytest.raises(ValueError, match="Blocked unsafe SQL"):
        execute_readonly_sql(db_file, "DROP TABLE sales")
