"""
Hybrid Retriever module.
Combines Vector Search (semantic) with BM25 (keyword search) using Reciprocal Rank Fusion (RRF).
"""

from rank_bm25 import BM25Okapi


class HybridSearcher:
    """Handles keyword search and fusion of search results."""

    def __init__(self, corpus_chunks: list[dict]):
        """Initializes the BM25 index with the given corpus.

        Args:
            corpus_chunks: List of dictionaries containing 'id' and 'text'.
        """
        self.corpus_chunks = corpus_chunks
        # Tokenize texts for BM25 (simple whitespace tokenization for speed)
        tokenized_corpus = [chunk["text"].lower().split() for chunk in corpus_chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search_bm25(self, query: str, top_k: int = 5) -> list[dict]:
        """Searches the corpus using BM25 keyword matching."""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        # Rank documents based on BM25 scores
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results = []
        for idx in ranked_indices[:top_k]:
            if scores[idx] > 0:  # Only include if there is some keyword match
                results.append(self.corpus_chunks[idx])
        return results

    @staticmethod
    def reciprocal_rank_fusion(
        vector_results: list[dict], bm25_results: list[dict], k: int = 60
    ) -> list[dict]:
        """Fuses results from Vector and BM25 searches using RRF.

        Args:
            vector_results: Ranked list of chunks from Vector DB.
            bm25_results: Ranked list of chunks from BM25.
            k: Smoothing constant for RRF.

        Returns:
            A single reranked list of chunks.
        """
        fused_scores = {}
        chunk_map = {}

        # Process Vector Results
        for rank, chunk in enumerate(vector_results):
            chunk_id = chunk.get("id", chunk["text"])  # Fallback to text if id is missing
            fused_scores[chunk_id] = 1 / (k + rank)
            chunk_map[chunk_id] = chunk

        # Process BM25 Results
        for rank, chunk in enumerate(bm25_results):
            chunk_id = chunk.get("id", chunk["text"])
            if chunk_id in fused_scores:
                fused_scores[chunk_id] += 1 / (k + rank)
            else:
                fused_scores[chunk_id] = 1 / (k + rank)
                chunk_map[chunk_id] = chunk

        # Sort by fused score descending
        sorted_fused = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

        # Return the actual chunk objects based on sorted order
        return [chunk_map[chunk_id] for chunk_id, score in sorted_fused]
