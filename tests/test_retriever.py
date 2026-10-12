"""
Unit tests for the RAG Retriever module.
"""

from unittest.mock import MagicMock

from src.rag.retriever import RAGRetriever


def test_get_relevant_context():
    """Test that query is properly embedded and searched in the vector store."""
    mock_embedder = MagicMock()
    # Mocking the deeply nested OpenAI response structure
    mock_response_data = MagicMock(embedding=[0.5, 0.6, 0.7])
    mock_embedder.client.embeddings.create.return_value.data = [mock_response_data]

    mock_vector_store = MagicMock()
    mock_vector_store.similarity_search.return_value = [
        {"id": "chunk_1", "text": "This is the answer", "metadata": {"source": "doc1.pdf"}},
    ]

    retriever = RAGRetriever(embedder=mock_embedder, vector_store=mock_vector_store)
    results = retriever.get_relevant_context("What is the answer?", top_k=1)

    assert len(results) == 1
    assert results[0]["text"] == "This is the answer"

    # Verify that the vector store was called with the exact mock embedding
    mock_vector_store.similarity_search.assert_called_once_with(
        query_embedding=[0.5, 0.6, 0.7], top_k=1
    )


def test_format_context():
    """Test that retrieved results are properly formatted into a single string."""
    retriever = RAGRetriever(embedder=MagicMock(), vector_store=MagicMock())

    sample_results = [
        {"text": "First chunk text", "metadata": {"source": "fileA.txt"}},
        {"text": "Second chunk text", "metadata": {"source": "fileB.txt"}},
    ]

    formatted_text = retriever.format_context(sample_results)

    assert "" in formatted_text
    assert "First chunk text" in formatted_text
    assert "" in formatted_text
    assert "Second chunk text" in formatted_text
