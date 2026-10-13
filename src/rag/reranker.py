"""
Reranker module for RAG pipeline using Cross-Encoders.
Re-scores retrieved context chunks to ensure top relevance before passing to the LLM.
"""

from typing import Any, Protocol

from src.utils.logger import logger


class CrossEncoderProtocol(Protocol):
    """Protocol defining expected CrossEncoder interface."""

    def predict(self, sentences: list[list[str]]) -> list[float]:
        """Predicts relevance scores for pairs of (query, doc)."""
        ...


class DocumentReranker:
    """Reranks retrieved document chunks using a Cross-Encoder model."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        model_instance: Any = None,
    ) -> None:
        """Initializes the reranker with a CrossEncoder model or injected mock instance.

        Args:
            model_name: HuggingFace model identifier for CrossEncoder.
            model_instance: Optional explicit CrossEncoder instance for testing.
        """
        self.model_name = model_name
        self._model = model_instance
        logger.info("DocumentReranker initialized with model: '%s'", self.model_name)

    @property
    def model(self) -> Any:
        """Lazy loading of the CrossEncoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name)
            except ImportError as e:
                logger.error("sentence_transformers or transformers library is missing.")
                raise RuntimeError(
                    "CrossEncoder requirements missing. Install transformers and sentence-transformers."
                ) from e
        return self._model

    def rerank(
        self, query: str, chunks: list[dict[str, Any]], top_k: int = 3
    ) -> list[dict[str, Any]]:
        """Reranks chunks based on fine-grained semantic relevance to the query.

        Args:
            query: User question or query string.
            chunks: List of document chunk dicts (must contain 'text' key).
            top_k: Number of top reranked chunks to return.

        Returns:
            List of top_k reranked chunks enriched with 'rerank_score'.
        """
        if not chunks or not query.strip():
            return chunks[:top_k]

        pairs = [[query, chunk.get("text", "")] for chunk in chunks]
        scores = self.model.predict(pairs)

        scored_chunks = []
        for score, chunk in zip(scores, chunks, strict=False):
            chunk_copy = dict(chunk)
            chunk_copy["rerank_score"] = float(score)
            scored_chunks.append(chunk_copy)

        scored_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
        logger.info("Reranked %d chunks down to top %d.", len(chunks), min(top_k, len(chunks)))

        return scored_chunks[:top_k]
