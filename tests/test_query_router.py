"""Tests for deterministic multi-source query router."""

from query_router import RouteTarget, route_query


def test_routes_sql_aggregation_queries():
    dec = route_query("What is the total revenue by region and top 5 customers in the sales table?")
    assert dec.target == RouteTarget.SQL
    assert dec.confidence >= 0.75


def test_routes_spreadsheet_queries():
    dec = route_query("Compare the Q3 budget column in the Excel sheet")
    assert dec.target == RouteTarget.SPREADSHEET
    assert dec.confidence >= 0.75


def test_routes_policy_questions_to_doc_rag():
    dec = route_query("What is our incident response SLA for P1 security breaches?")
    assert dec.target == RouteTarget.DOC_RAG


def test_falls_back_to_doc_rag_when_sql_disabled():
    dec = route_query("Show top 5 rows from table", has_sql_db=False, has_spreadsheet=False)
    assert dec.target == RouteTarget.DOC_RAG
