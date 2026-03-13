"""
sql_agent.py — Natural language → SQLite queries using LangChain + Ollama.

Two entry points:
1. query_sql(db_path, question)  — query an existing SQLite .db file
2. excel_to_sqlite(excel_path, db_path)  — convert Excel → SQLite, then query

The chain first generates a SQL query, executes it, then explains the result
in plain language.
"""

import os
import sqlite3
from pathlib import Path

import pandas as pd
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_ollama import ChatOllama

from config import LLM_MODEL, OLLAMA_BASE_URL


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------

def excel_to_sqlite(excel_path: str, db_path: str) -> list[str]:
    """
    Convert all sheets of an Excel file into SQLite tables.

    Returns a list of table names created.
    """
    ext = Path(excel_path).suffix.lower()
    conn = sqlite3.connect(db_path)
    tables = []

    try:
        if ext == ".csv":
            df = pd.read_csv(excel_path)
            table_name = Path(excel_path).stem.replace(" ", "_").replace("-", "_")
            df.to_sql(table_name, conn, if_exists="replace", index=False)
            tables.append(table_name)
        else:
            xls = pd.ExcelFile(excel_path)
            for sheet in xls.sheet_names:
                df = xls.parse(sheet)
                df = df.dropna(how="all").dropna(axis=1, how="all")
                if df.empty:
                    continue
                table_name = sheet.replace(" ", "_").replace("-", "_")
                df.to_sql(table_name, conn, if_exists="replace", index=False)
                tables.append(table_name)
    finally:
        conn.close()

    return tables


def get_tables(db_path: str) -> list[str]:
    """List all table names in a SQLite database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    return tables


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

def query_sql(db_path: str, question: str) -> dict:
    """
    Translate a natural-language question into SQL, run it, and explain the result.

    Returns:
        {
            "sql_query": str,
            "raw_result": str,
            "answer": str,
            "tables": list[str]
        }
    """
    db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
    tables = get_tables(db_path)

    llm = ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
    )

    # Step 1: Generate SQL
    sql_chain = create_sql_query_chain(llm, db)
    sql_query = sql_chain.invoke({"question": question})

    # Clean up potential markdown wrapping from the LLM
    sql_query = sql_query.strip()
    if sql_query.startswith("```"):
        lines = sql_query.split("\n")
        sql_query = "\n".join(
            line for line in lines
            if not line.startswith("```") and not line.lower().startswith("sql")
        ).strip()

    # Step 2: Execute SQL
    try:
        raw_result = db.run(sql_query)
    except Exception as e:
        return {
            "sql_query": sql_query,
            "raw_result": f"Execution error: {e}",
            "answer": f"The generated SQL query could not be executed: {e}",
            "tables": tables,
        }

    # Step 3: Explain result in plain language
    explain_prompt = (
        f"The user asked: '{question}'\n"
        f"SQL query used: {sql_query}\n"
        f"Query result: {raw_result}\n\n"
        "Explain this result clearly in 1-3 sentences, as if talking to a non-technical user."
    )
    explanation = llm.invoke(explain_prompt).content

    return {
        "sql_query": sql_query,
        "raw_result": str(raw_result),
        "answer": explanation,
        "tables": tables,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python sql_agent.py <database.db> <question>")
        sys.exit(1)

    result = query_sql(sys.argv[1], " ".join(sys.argv[2:]))
    print(f"\nSQL: {result['sql_query']}")
    print(f"Raw:  {result['raw_result']}")
    print(f"\nAnswer: {result['answer']}")
