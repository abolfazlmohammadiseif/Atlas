"""
Retriever module for RAG pipeline.
Converts user queries to embeddings and fetches relevant context from the vector store.
"""

from typing import Any

from src.rag.embedder import EmbeddingClient
from src.rag.vector_store import VectorStoreManager
from src.utils.logger import logger


class RAGRetriever:
    """Orchestrates query embedding and vector database retrieval."""

    def __init__(self, embedder: EmbeddingClient, vector_store: VectorStoreManager) -> None:
        """Initializes the retriever with an embedder and a vector store."""
        self.embedder = embedder
        self.vector_store = vector_store
        logger.info("RAG Retriever initialized.")

    def get_relevant_context(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Retrieves top-k relevant document chunks for a given natural language query.

        Args:
            query: The user's question.
            top_k: Number of most relevant chunks to return.

        Returns:
            List of dictionaries containing text, metadata, and distance.
        """
        # 1. Convert user query into a vector embedding
        response = self.embedder.client.embeddings.create(
            model=self.embedder.model,
            input=query,
        )
        query_embedding = response.data[0].embedding

        # 2. Search the vector store for closest chunks
        results = self.vector_store.similarity_search(query_embedding=query_embedding, top_k=top_k)

        logger.info("Retrieved %d context chunks for query: '%s'", len(results), query)
        return results

    def format_context(self, results: list[dict[str, Any]]) -> str:
        """Formats the retrieved dictionary chunks into a single readable string for the LLM.

        Args:
            results: The output from get_relevant_context.

        Returns:
            A formatted string ready to be injected into an LLM prompt.
        """
        context_parts = []
        for result in results:
            result.get("metadata", {}).get("source", "Unknown Source")
            text = result.get("text", "")
            context_parts.append(f"\n{text}\n")

        return "\n".join(context_parts)
