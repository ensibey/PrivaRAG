"""Natural language to SQLite query execution via LangChain and local Ollama."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, List

import pandas as pd
from langchain.chains import create_sql_query_chain
from langchain_community.utilities import SQLDatabase
from langchain_ollama import ChatOllama

from config import LLM_MODEL, OLLAMA_BASE_URL


def excel_to_sqlite(excel_path: str, db_path: str) -> List[str]:
    """Load sheets from an Excel or CSV file into SQLite tables."""
    ext = Path(excel_path).suffix.lower()
    conn = sqlite3.connect(db_path)
    tables: List[str] = []

    try:
        if ext == ".csv":
            df = pd.read_csv(excel_path)
            table_name = Path(excel_path).stem.replace(" ", "_").replace("-", "_")
            df.to_sql(table_name, conn, if_exists="replace", index=False)
            tables.append(table_name)
        else:
            xls = pd.ExcelFile(excel_path)
            for sheet in xls.sheet_names:
                df = xls.parse(sheet).dropna(how="all").dropna(axis=1, how="all")
                if df.empty:
                    continue
                table_name = sheet.replace(" ", "_").replace("-", "_")
                df.to_sql(table_name, conn, if_exists="replace", index=False)
                tables.append(table_name)
    finally:
        conn.close()

    return tables


def get_tables(db_path: str) -> List[str]:
    """List user tables in a SQLite database."""
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        return [row[0] for row in cursor.fetchall()]
    finally:
        conn.close()


def query_sql(db_path: str, question: str) -> Dict[str, object]:
    """Generate a SQL query from natural language, run it against SQLite, and summarize."""
    db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
    tables = get_tables(db_path)

    llm = ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
    )

    sql_chain = create_sql_query_chain(llm, db)
    sql_query = sql_chain.invoke({"question": question}).strip()

    if sql_query.startswith("```"):
        lines = sql_query.split("\n")
        sql_query = "\n".join(
            line
            for line in lines
            if not line.startswith("```") and not line.lower().startswith("sql")
        ).strip()

    try:
        raw_result = db.run(sql_query)
    except Exception as exc:
        return {
            "sql_query": sql_query,
            "raw_result": f"Execution error: {exc}",
            "answer": f"Generated SQL query failed to execute: {exc}",
            "tables": tables,
        }

    explain_prompt = (
        f"Question: '{question}'\n"
        f"SQL query executed: {sql_query}\n"
        f"Raw result: {raw_result}\n\n"
        "Summarize the result concisely in 1-3 sentences."
    )
    explanation = llm.invoke(explain_prompt).content

    return {
        "sql_query": sql_query,
        "raw_result": str(raw_result),
        "answer": explanation,
        "tables": tables,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python sql_agent.py <database.db> <question>")
        sys.exit(1)

    res = query_sql(sys.argv[1], " ".join(sys.argv[2:]))
    print(f"SQL:    {res['sql_query']}")
    print(f"Result: {res['raw_result']}")
    print(f"Answer: {res['answer']}")
