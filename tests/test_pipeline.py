"""
Unit tests for the complete RAG Pipeline.
"""

from unittest.mock import patch

from src.rag.pipeline import RAGPipeline


@patch("src.rag.pipeline.DocumentProcessor")
@patch("src.rag.pipeline.EmbeddingClient")
@patch("src.rag.pipeline.VectorStoreManager")
@patch("src.rag.pipeline.RAGRetriever")
@patch("src.rag.pipeline.RAGGenerator")
def test_pipeline_ingest_text(mock_gen, mock_ret, mock_vec, mock_emb, mock_doc):
    """Test the data ingestion flow (Chunk -> Embed -> Store)."""
    # Create the pipeline with mocked components
    pipeline = RAGPipeline()

    # Mock return values for the sequence
    pipeline.processor.process_text.return_value = ["chunk1", "chunk2"]
    pipeline.embedder.process_chunks.return_value = ["vector1", "vector2"]

    # Run the method
    pipeline.ingest_text("Sample long text", metadata={"source": "test"})

    # Assertions to ensure sequence was followed
    pipeline.processor.process_text.assert_called_once_with(
        "Sample long text", metadata={"source": "test"}
    )
    pipeline.embedder.process_chunks.assert_called_once_with(["chunk1", "chunk2"])
    pipeline.vector_store.add_vector_chunks.assert_called_once_with(["vector1", "vector2"])


@patch("src.rag.pipeline.DocumentProcessor")
@patch("src.rag.pipeline.EmbeddingClient")
@patch("src.rag.pipeline.VectorStoreManager")
@patch("src.rag.pipeline.RAGRetriever")
@patch("src.rag.pipeline.RAGGenerator")
def test_pipeline_ask(mock_gen, mock_ret, mock_vec, mock_emb, mock_doc):
    """Test the question answering flow (Retrieve -> Format -> Generate)."""
    pipeline = RAGPipeline()

    # Mock return values
    pipeline.retriever.get_relevant_context.return_value = [{"text": "context snippet"}]
    pipeline.retriever.format_context.return_value = "Formatted Context"
    pipeline.generator.generate_answer.return_value = "Final Answer"

    # Run the method
    answer = pipeline.ask("What is RAG?")

    # Assertions
    assert answer == "Final Answer"
    pipeline.retriever.get_relevant_context.assert_called_once_with(query="What is RAG?", top_k=3)
    pipeline.retriever.format_context.assert_called_once_with([{"text": "context snippet"}])
    pipeline.generator.generate_answer.assert_called_once_with(
        query="What is RAG?", context="Formatted Context"
    )
