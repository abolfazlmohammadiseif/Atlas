"""
Vector Store Management Engine using ChromaDB.
Handles storing, indexing, and similarity searching of vector chunks.
"""

from typing import Any

import chromadb

from src.rag.embedder import VectorChunk
from src.utils.logger import logger


class VectorStoreManager:
    """Manages local vector database operations for RAG retrieval."""

    def __init__(
        self, collection_name: str = "research_documents", persist_directory: str = "./chroma_db"
    ) -> None:
        """Initializes ChromaDB persistent client and collection.

        Args:
            collection_name: Name of the vector collection.
            persist_directory: Path to store vector database files locally.
        """
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        logger.info(
            "Initialized ChromaDB collection '%s' at '%s'", collection_name, persist_directory
        )

    def add_vector_chunks(self, vector_chunks: list[VectorChunk]) -> None:
        """Adds a list of VectorChunks into the vector database.

        Args:
            vector_chunks: List of embedded chunks to store.
        """
        if not vector_chunks:
            return

        ids = [chunk.chunk_id for chunk in vector_chunks]
        embeddings = [chunk.embedding for chunk in vector_chunks]
        documents = [chunk.text for chunk in vector_chunks]
        metadatas = [chunk.metadata for chunk in vector_chunks]

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info("Successfully indexed %d chunks into Vector Store.", len(vector_chunks))

    def similarity_search(
        self, query_embedding: list[float], top_k: int = 3
    ) -> list[dict[str, Any]]:
        """Searches for the most similar chunks given a query embedding.

        Args:
            query_embedding: The vector embedding of the user query.
            top_k: Number of top results to return.

        Returns:
            list[dict]: Retrieved chunks with text, metadata, and distance scores.
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        formatted_results = []
        if results and results["documents"]:
            for i in range(len(results["documents"][0])):
                formatted_results.append(
                    {
                        "id": results["ids"][0][i],
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i] if "distances" in results else None,
                    }
                )

        logger.info("Retrieved %d top matches from vector store.", len(formatted_results))
        return formatted_results


if __name__ == "__main__":
    # Test script for VectorStoreManager
    store = VectorStoreManager(collection_name="test_collection")
    print("VectorStoreManager initialized successfully.")
