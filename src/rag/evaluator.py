"""
Evaluation lab for RAG retrieval performance.
Calculates information retrieval metrics like NDCG@K and MRR (Mean Reciprocal Rank).
"""

import math


class RetrievalEvaluator:
    """Computes ranking evaluation metrics for retrieved document chunks."""

    @staticmethod
    def calculate_mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
        """Calculates Mean Reciprocal Rank (MRR) for a single query.

        Args:
            retrieved_ids: Ranked list of document IDs returned by retriever.
            relevant_ids: Set of ground-truth relevant document IDs.

        Returns:
            Reciprocal rank value (1/rank) or 0.0 if no relevant doc found.
        """
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant_ids:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def calculate_ndcg(
        retrieved_ids: list[str], relevance_scores: dict[str, int], k: int = 5
    ) -> float:
        """Calculates Normalized Discounted Cumulative Gain (NDCG@K).

        Args:
            retrieved_ids: Ranked list of document IDs returned by retriever.
            relevance_scores: Mapping of doc_id to integer relevance score (e.g., 0 to 3).
            k: Top-K evaluation cutoff depth.

        Returns:
            NDCG score between 0.0 and 1.0.
        """
        top_k_ids = retrieved_ids[:k]

        # Calculate DCG@K
        dcg = 0.0
        for rank, doc_id in enumerate(top_k_ids, start=1):
            rel = relevance_scores.get(doc_id, 0)
            if rel > 0:
                dcg += (2**rel - 1) / math.log2(rank + 1)

        # Calculate Ideal DCG@K (IDCG)
        sorted_relevances = sorted(relevance_scores.values(), reverse=True)[:k]
        idcg = 0.0
        for rank, rel in enumerate(sorted_relevances, start=1):
            if rel > 0:
                idcg += (2**rel - 1) / math.log2(rank + 1)

        if idcg == 0.0:
            return 0.0

        return dcg / idcg
