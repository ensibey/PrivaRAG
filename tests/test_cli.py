"""Tests for standalone offline PrivaRAG CLI."""

import json
from pathlib import Path

from cli import main


def test_cli_search_redact_sql_and_route(tmp_path: Path, capsys):
    (tmp_path / "sec.md").write_text("Air-gapped Ollama runs completely offline with zero data egress.", encoding="utf-8")
    (tmp_path / "hr.md").write_text("Parental leave policy provides 16 weeks paid leave.", encoding="utf-8")

    assert main(["search", str(tmp_path), "offline air-gapped Ollama", "--json"]) == 0
    search_out = json.loads(capsys.readouterr().out)
    assert search_out["files_indexed"] == 2
    assert "sec.md" in search_out["hits"][0]["source"]

    assert main(["redact", "Mail: test@corp.com"]) == 0
    redact_out = json.loads(capsys.readouterr().out)
    assert "[REDACTED_EMAIL]" in redact_out["redacted_text"]

    assert main(["audit-sql", "SELECT * FROM sales"]) == 0
    capsys.readouterr()
    assert main(["audit-sql", "DROP TABLE sales"]) == 1
    capsys.readouterr()

    assert main(["route", "Show total revenue in Excel sheet"]) == 0
    route_out = json.loads(capsys.readouterr().out)
    assert route_out["target"] == "SPREADSHEET"
