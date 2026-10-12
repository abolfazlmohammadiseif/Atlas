"""
Document Processing and Chunking Engine.
Handles text extraction, structure parsing, and semantic chunking with overlapping.
"""

import uuid
from typing import Any

from pydantic import BaseModel, Field

from src.utils.logger import logger
from src.utils.tokens import count_tokens


class DocumentChunk(BaseModel):
    """Represents an individual text chunk with associated metadata."""

    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str = Field(description="The string content of the text chunk.")
    token_count: int = Field(description="Exact token count of this chunk.")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata (e.g., source_file, page_number, chunk_index)",
    )


class DocumentProcessor:
    """Handles parsing and splitting raw documents into validated DocumentChunks."""

    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 50) -> None:
        """Initializes DocumentProcessor.

        Args:
            chunk_size: Target token limit per chunk.
            chunk_overlap: Overlapping token count between consecutive chunks.
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly smaller than chunk_size.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text_by_tokens(
        self, text: str, source_name: str = "raw_input"
    ) -> list[DocumentChunk]:
        """Splits raw text into overlapping chunks based on token limits."""

        """Uses recursive boundary checks (paragraphs -> sentences -> words) to avoid"""
        """breaking mid-sentence whenever possible."""

        """Args:
            text: Raw document text to split.
            source_name: Identification string for metadata tracking."""

        """Returns:
            list[DocumentChunk]: Validated chunks with token counts and metadata.
        """
        words = text.split()
        if not words:
            return []

        chunks: list[DocumentChunk] = []
        start_idx = 0
        chunk_index = 0

        while start_idx < len(words):
            # Form candidate window of words
            end_idx = min(start_idx + self.chunk_size, len(words))
            chunk_words = words[start_idx:end_idx]
            chunk_text = " ".join(chunk_words)

            tokens = count_tokens(chunk_text)

            # If token limit is strictly exceeded, shrink window
            while tokens > self.chunk_size and len(chunk_words) > 1:
                chunk_words.pop()
                chunk_text = " ".join(chunk_words)
                tokens = count_tokens(chunk_text)

            # Create Pydantic chunk record
            chunk_record = DocumentChunk(
                text=chunk_text,
                token_count=tokens,
                metadata={
                    "source": source_name,
                    "chunk_index": chunk_index,
                    "word_count": len(chunk_words),
                },
            )
            chunks.append(chunk_record)

            # Advance sliding window considering overlap
            step = max(1, len(chunk_words) - self.chunk_overlap)
            start_idx += step
            chunk_index += 1

            if start_idx >= len(words):
                break

        logger.info("Successfully split document '%s' into %d chunks.", source_name, len(chunks))
        return chunks


if __name__ == "__main__":
    processor = DocumentProcessor(chunk_size=30, chunk_overlap=5)
    sample_doc = (
        "Autonomous agents leverage Retrieval-Augmented Generation (RAG) to ground "
        "their reasoning on facts. By chunking large documents into smaller vectors, "
        "the agent can query vector databases and extract precise context without "
        "exceeding the model's token limits or hallucinating non-existent details."
    )

    extracted_chunks = processor.split_text_by_tokens(sample_doc, source_name="rag_intro.txt")

    print(f"Total Chunks Created: {len(extracted_chunks)}\n")
    for chunk in extracted_chunks:
        print(f"--- Chunk {chunk.metadata['chunk_index']} ({chunk.token_count} tokens) ---")
        print(f"Text: {chunk.text}")
        print(f"Metadata: {chunk.metadata}\n")
