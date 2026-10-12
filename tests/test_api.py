"""
Unit tests for the FastAPI application.
"""

from unittest.mock import patch

# Patch OpenAI BEFORE importing app/pipeline to avoid missing API key errors on collection
with patch("src.rag.embedder.OpenAI"), patch("src.rag.generator.OpenAI"):
    from fastapi.testclient import TestClient

    from src.api.main import app, pipeline

client = TestClient(app)


@patch.object(pipeline, "ingest_text")
def test_ingest_endpoint(mock_ingest):
    """Test the /ingest endpoint successfully processes requests."""
    response = client.post(
        "/ingest",
        json={"text": "FastAPI is great for building APIs.", "metadata": {"source": "web"}},
    )

    assert response.status_code == 200
    assert response.json() == {"status": "success", "message": "Document ingested successfully."}

    mock_ingest.assert_called_once_with(
        text="FastAPI is great for building APIs.", metadata={"source": "web"}
    )


@patch.object(pipeline, "ask")
def test_ask_endpoint(mock_ask):
    """Test the /ask endpoint successfully returns an answer."""
    mock_ask.return_value = "This is a mock answer from the LLM."

    response = client.post("/ask", json={"question": "What is FastAPI?"})

    assert response.status_code == 200
    assert response.json() == {"answer": "This is a mock answer from the LLM."}

    mock_ask.assert_called_once_with(question="What is FastAPI?")
