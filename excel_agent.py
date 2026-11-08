"""Spreadsheet analysis agent backed by Pandas and local Ollama."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_ollama import ChatOllama

from config import LLM_MODEL, OLLAMA_BASE_URL


def load_dataframe(file_path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """Load an Excel or CSV file into a cleaned Pandas DataFrame."""
    ext = Path(file_path).suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(file_path)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(file_path, sheet_name=sheet_name or 0)
    else:
        raise ValueError(f"Unsupported format: {ext}. Expected .xlsx, .xls, or .csv")

    return df.dropna(how="all").dropna(axis=1, how="all")


def get_sheet_names(file_path: str) -> List[str]:
    """Return sheet names for Excel workbooks or ['CSV'] for CSV files."""
    if Path(file_path).suffix.lower() == ".csv":
        return ["CSV"]
    xls = pd.ExcelFile(file_path)
    return list(xls.sheet_names)


def query_excel(
    file_path: str,
    question: str,
    sheet_name: Optional[str] = None,
) -> Dict[str, str]:
    """Execute a natural-language analytical query over a spreadsheet."""
    df = load_dataframe(file_path, sheet_name)

    llm = ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
    )

    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=False,
        allow_dangerous_code=True,
        agent_type="tool-calling",
    )

    answer = agent.invoke({"input": question})
    answer_text = answer.get("output", str(answer))
    info = f"Rows: {len(df):,} | Columns: {', '.join(str(c) for c in df.columns)}"

    return {
        "answer": answer_text,
        "dataframe_info": info,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python excel_agent.py <file.xlsx> <question>")
        sys.exit(1)

    res = query_excel(sys.argv[1], " ".join(sys.argv[2:]))
    print(f"Answer: {res['answer']}")
    print(f"Schema: {res['dataframe_info']}")
