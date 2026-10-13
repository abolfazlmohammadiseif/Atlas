"""
Unit tests for the DocumentReranker module using Dependency Injection.
"""

from unittest.mock import MagicMock

from src.rag.reranker import DocumentReranker


def test_rerank_chunks():
    """Test that chunks are re-scored and sorted strictly by score."""
    mock_model = MagicMock()
    # Simulate scores: chunk 2 gets highest score (0.92), then chunk 3 (0.65), then chunk 1 (0.15)
    mock_model.predict.return_value = [0.15, 0.92, 0.65]

    # Inject mock model directly to avoid sentence_transformers import during unit test
    reranker = DocumentReranker(model_instance=mock_model)

    sample_chunks = [
        {"id": "doc_1", "text": "Low relevance text"},
        {"id": "doc_2", "text": "High relevance text"},
        {"id": "doc_3", "text": "Medium relevance text"},
    ]

    results = reranker.rerank(query="What is high relevance?", chunks=sample_chunks, top_k=2)

    assert len(results) == 2
    assert results[0]["id"] == "doc_2"
    assert results[0]["rerank_score"] == 0.92
    assert results[1]["id"] == "doc_3"
    assert results[1]["rerank_score"] == 0.65


def test_rerank_empty_chunks():
    """Test reranker behavior with empty chunks input."""
    mock_model = MagicMock()
    reranker = DocumentReranker(model_instance=mock_model)
    results = reranker.rerank(query="Test query", chunks=[], top_k=3)
    assert results == []
