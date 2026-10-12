"""
Unit tests for vector database store operations.
"""

from unittest.mock import MagicMock, patch

from src.rag.vector_store import VectorStoreManager


@patch("src.rag.vector_store.chromadb.PersistentClient")
def test_vector_store_initialization(mock_client_class):
    """Test that ChromaDB client and collection are properly initialized."""
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    VectorStoreManager(collection_name="my_test_col")

    mock_client_class.assert_called_once()
    mock_client.get_or_create_collection.assert_called_once_with(name="my_test_col")


@patch("src.rag.vector_store.chromadb.PersistentClient")
def test_similarity_search(mock_client_class):
    """Test searching the vector store returns formatted results."""
    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection
    mock_client_class.return_value = mock_client

    # Mock ChromaDB query output
    mock_collection.query.return_value = {
        "ids": [["id1"]],
        "documents": [["Sample text match"]],
        "metadatas": [[{"source": "test.txt"}]],
        "distances": [[0.15]],
    }

    store = VectorStoreManager()
    results = store.similarity_search([0.1, 0.2, 0.3], top_k=1)

    assert len(results) == 1
    assert results[0]["id"] == "id1"
    assert results[0]["text"] == "Sample text match"
    assert results[0]["metadata"]["source"] == "test.txt"
    assert results[0]["distance"] == 0.15
