"""
End-to-End RAG Pipeline Orchestrator.
Ties together chunking, embedding, vector storage, retrieval, and generation.
"""

from typing import Any

from src.rag.document_processor import DocumentProcessor
from src.rag.embedder import EmbeddingClient
from src.rag.generator import RAGGenerator
from src.rag.retriever import RAGRetriever
from src.rag.vector_store import VectorStoreManager
from src.utils.logger import logger


class RAGPipeline:
    """Orchestrates the entire RAG workflow from ingestion to generation."""

    def __init__(self) -> None:
        """Initializes all components of the RAG pipeline."""
        self.processor = DocumentProcessor()
        self.embedder = EmbeddingClient()
        self.vector_store = VectorStoreManager()

        self.retriever = RAGRetriever(embedder=self.embedder, vector_store=self.vector_store)
        self.generator = RAGGenerator()

        logger.info("End-to-End RAG Pipeline initialized successfully.")

    def ingest_text(self, text: str, metadata: dict[str, Any] | None = None) -> None:
        """Processes raw text and indexes it into the vector database.

        Args:
            text: The raw document text to ingest.
            metadata: Optional dictionary of metadata (e.g., source, author).
        """
        if not text.strip():
            logger.warning("Empty text provided for ingestion. Skipping.")
            return

        # 1. Chunk the document
        chunks = self.processor.process_text(text, metadata=metadata or {})

        # 2. Embed the chunks
        vector_chunks = self.embedder.process_chunks(chunks)

        # 3. Store in Vector DB
        self.vector_store.add_vector_chunks(vector_chunks)

        logger.info("Successfully ingested document resulting in %d chunks.", len(vector_chunks))

    def ask(self, question: str) -> str:
        """Answers a user's question based on ingested documents.

        Args:
            question: The user's query.

        Returns:
            The generated answer from the LLM.
        """
        # 1. Retrieve relevant chunks
        results = self.retriever.get_relevant_context(query=question, top_k=3)

        # 2. Format context
        context = self.retriever.format_context(results)

        if not context.strip():
            logger.warning("No relevant context found for the query.")
            return "I'm sorry, but I couldn't find any relevant information \
                 in the documents to answer your question."

        # 3. Generate answer
        answer = self.generator.generate_answer(query=question, context=context)

        return answer
