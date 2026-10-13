"""
Unit tests for the embedding engine.
"""

from unittest.mock import MagicMock, patch

from src.rag.document_processor import DocumentChunk
from src.rag.embedder import EmbeddingClient


@patch("src.rag.embedder.OpenAI")
def test_process_chunks_success(mock_openai_class):
    """Test that chunks are properly embedded using a mocked OpenAI client."""
    mock_client_instance = MagicMock()
    mock_openai_class.return_value = mock_client_instance

    mock_response = MagicMock()
    mock_response.data = [
        MagicMock(embedding=[0.1, 0.2, 0.3]),
        MagicMock(embedding=[0.4, 0.5, 0.6]),
    ]
    mock_client_instance.embeddings.create.return_value = mock_response

    chunks = [
        DocumentChunk(text="Chunk one", token_count=2, metadata={}),
        DocumentChunk(text="Chunk two", token_count=2, metadata={}),
    ]

    embedder = EmbeddingClient()
    vector_chunks = embedder.process_chunks(chunks)

    assert len(vector_chunks) == 2
    assert vector_chunks[0].embedding == [0.1, 0.2, 0.3]
    assert vector_chunks[1].embedding == [0.4, 0.5, 0.6]

    mock_client_instance.embeddings.create.assert_called_once()


@patch("src.rag.embedder.OpenAI")
def test_process_chunks_empty_list(mock_openai_class):
    """Test that an empty list returns an empty list without calling API."""
    # Even though we don't call the API, we need to mock it so __init__ doesn't fail
    embedder = EmbeddingClient()

    # Should return immediately without initializing API call
    result = embedder.process_chunks([])
    assert result == []

    # Verify the API was NEVER called
    mock_client_instance = mock_openai_class.return_value
    mock_client_instance.embeddings.create.assert_not_called()
