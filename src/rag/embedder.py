"""
Embedding Engine for RAG.
Converts text chunks into semantic vector embeddings using OpenAI API.
"""

from openai import OpenAI
from pydantic import Field

from src.rag.document_processor import DocumentChunk
from src.utils.logger import logger


class VectorChunk(DocumentChunk):
    """A document chunk that includes its vector representation."""

    embedding: list[float] = Field(
        description="The numerical vector representation of the chunk text."
    )


class EmbeddingClient:
    """Handles communication with embedding models for vector generation."""

    def __init__(self, model_name: str = "text-embedding-3-small") -> None:
        """Initializes the EmbeddingClient.

        Args:
            model_name: The OpenAI embedding model to use.
        """
        # Automatically loads OPENAI_API_KEY from environment variables
        self.client = OpenAI()
        self.model_name = model_name

    def process_chunks(self, chunks: list[DocumentChunk]) -> list[VectorChunk]:
        """Converts a list of DocumentChunks into VectorChunks using batch API.

        Args:
            chunks: List of DocumentChunks extracted from text.

        Returns:
            list[VectorChunk]: The chunks enriched with their embeddings.
        """
        if not chunks:
            return []

        logger.info("Generating embeddings for %d chunks...", len(chunks))

        # Replace newlines with spaces as recommended by OpenAI for better embeddings
        texts = [chunk.text.replace("\n", " ") for chunk in chunks]

        vector_chunks = []
        try:
            # Batch request: send all chunks in one API call to save time and network overhead
            response = self.client.embeddings.create(input=texts, model=self.model_name)

            for index, chunk in enumerate(chunks):
                vector_chunks.append(
                    VectorChunk(
                        # Unpack existing DocumentChunk data (text, metadata, etc.)
                        **chunk.model_dump(),
                        # Add the newly generated embedding
                        embedding=response.data[index].embedding,
                    )
                )

            logger.info("Successfully generated embeddings for all chunks.")
            return vector_chunks

        except Exception as e:
            logger.error("Failed to generate embeddings: %s", e)
            raise


if __name__ == "__main__":
    from src.rag.document_processor import DocumentProcessor

    # 1. Create a dummy chunk using our previous module
    processor = DocumentProcessor(chunk_size=50, chunk_overlap=10)
    sample_text = "Machine learning engineers build scalable AI systems."
    chunks = processor.split_text_by_tokens(sample_text, source_name="test")

    # 2. Test the embedder
    embedder = EmbeddingClient()
    try:
        vectorized_chunks = embedder.process_chunks(chunks)
        print(f"Original Text: {vectorized_chunks[0].text}")
        print(f"Embedding Dimension: {len(vectorized_chunks[0].embedding)}")
        print(f"First 5 values of embedding: {vectorized_chunks[0].embedding[:5]}")
    except Exception as err:
        print(f"API Error (Ensure OPENAI_API_KEY is set): {err}")
