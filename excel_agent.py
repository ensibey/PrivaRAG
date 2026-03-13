"""
excel_agent.py — Natural language analysis of Excel/CSV files via Pandas.

Uses LangChain's PandasDataFrameAgent powered by local Ollama llama3.
The agent can compute sums, averages, filters, and comparisons directly
on the DataFrame without needing a vector database.
"""

import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_ollama import ChatOllama

from config import LLM_MODEL, OLLAMA_BASE_URL


def load_dataframe(file_path: str, sheet_name: str | None = None) -> pd.DataFrame:
    """Load an Excel or CSV file into a Pandas DataFrame."""
    import re
    from pathlib import Path

    ext = Path(file_path).suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(file_path)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(file_path, sheet_name=sheet_name or 0)
    else:
        raise ValueError(f"Unsupported format: {ext}. Use .xlsx, .xls, or .csv")

    df = df.dropna(how="all").dropna(axis=1, how="all")
    return df


def get_sheet_names(file_path: str) -> list[str]:
    """Return sheet names for Excel files, or ['CSV'] for CSV."""
    from pathlib import Path
    if Path(file_path).suffix.lower() == ".csv":
        return ["CSV"]
    xls = pd.ExcelFile(file_path)
    return xls.sheet_names


def query_excel(file_path: str, question: str, sheet_name: str | None = None) -> dict:
    """
    Ask a natural-language question about an Excel/CSV file.

    Returns:
        {"answer": str, "dataframe_info": str}
    """
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

    # Build a quick summary of the DataFrame for context display
    info_lines = [
        f"Rows: {len(df):,}",
        f"Columns: {', '.join(df.columns.tolist())}",
    ]

    return {
        "answer": answer_text,
        "dataframe_info": " | ".join(info_lines),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python excel_agent.py <file.xlsx> <question>")
        sys.exit(1)

    result = query_excel(sys.argv[1], " ".join(sys.argv[2:]))
    print(f"\nAnswer: {result['answer']}")
    print(f"File info: {result['dataframe_info']}")
