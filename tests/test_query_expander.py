"""
Unit tests for the QueryExpander module.
"""

from unittest.mock import MagicMock, patch

from src.rag.query_expander import QueryExpander


@patch("src.rag.query_expander.OpenAI")
def test_expand_query_success(mock_openai_class):
    """Test that a query is successfully expanded into multiple variations."""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client

    mock_response = MagicMock()
    mock_response.choices[
        0
    ].message.content = "What are modern python web frameworks?\nBest python web tools"
    mock_client.chat.completions.create.return_value = mock_response

    expander = QueryExpander()
    queries = expander.expand_query("What is FastAPI?", num_queries=2)

    assert len(queries) == 3
    assert queries[0] == "What is FastAPI?"
    assert "What are modern python web frameworks?" in queries
    assert "Best python web tools" in queries


@patch("src.rag.query_expander.OpenAI")
def test_expand_query_empty_string(mock_openai_class):
    """Test that expanding an empty query string returns an empty list."""
    expander = QueryExpander()
    assert expander.expand_query("") == []
