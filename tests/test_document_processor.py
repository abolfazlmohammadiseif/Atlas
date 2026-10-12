"""
Unit tests for document chunking logic.
"""

import pytest
from src.rag.document_processor import DocumentProcessor

def test_document_processor_invalid_overlap():
    """Ensure initializing with overlap >= size raises ValueError."""
    with pytest.raises(ValueError, match="chunk_overlap must be strictly smaller"):
        DocumentProcessor(chunk_size=100, chunk_overlap=100)

def test_split_text_empty():
    """Ensure empty string returns empty chunk list."""
    processor = DocumentProcessor()
    assert processor.split_text_by_tokens("") == []

def test_split_text_chunks_created():
    """Ensure text is properly split into chunks with metadata."""
    processor = DocumentProcessor(chunk_size=15, chunk_overlap=3)
    text = "The quick brown fox jumps over the lazy dog multiple times to create a longer sentence for testing."
    chunks = processor.split_text_by_tokens(text, source_name="test_doc")

    assert len(chunks) > 1
    assert chunks[0].metadata["source"] == "test_doc"
    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[0].token_count <= 15

