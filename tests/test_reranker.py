"""
Unit tests for the DocumentReranker module.
"""

from unittest.mock import MagicMock, patch

from src.rag.reranker import DocumentReranker


@patch("src.rag.reranker.CrossEncoder")
def test_rerank_chunks(mock_cross_encoder_class):
    """Test that chunks are re-scored and sorted strictly by score."""
    mock_model = MagicMock()
    mock_cross_encoder_class.return_value = mock_model

    # Simulate scores: chunk 2 gets highest score, followed by chunk 3 and chunk 1
    mock_model.predict.return_value = [0.15, 0.92, 0.65]

    reranker = DocumentReranker()

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


@patch("src.rag.reranker.CrossEncoder")
def test_rerank_empty_chunks(mock_cross_encoder_class):
    """Test reranker behavior with empty chunks input."""
    reranker = DocumentReranker()
    results = reranker.rerank(query="Test query", chunks=[], top_k=3)
    assert results == []
