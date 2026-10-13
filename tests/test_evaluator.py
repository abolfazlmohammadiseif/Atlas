"""
Unit tests for RetrievalEvaluator metrics (MRR and NDCG).
"""

from src.rag.evaluator import RetrievalEvaluator


def test_calculate_mrr():
    """Test MRR calculation for relevant document at rank 2."""
    retrieved = ["doc_A", "doc_B", "doc_C"]
    relevant = {"doc_B"}

    mrr = RetrievalEvaluator.calculate_mrr(retrieved, relevant)
    assert mrr == 0.5  # 1 / 2nd position


def test_calculate_ndcg_perfect_order():
    """Test NDCG@K returns 1.0 when retrieved documents are in ideal relevance order."""
    retrieved = ["doc_1", "doc_2", "doc_3"]
    relevance_scores = {"doc_1": 3, "doc_2": 2, "doc_3": 1}

    ndcg = RetrievalEvaluator.calculate_ndcg(retrieved, relevance_scores, k=3)
    assert round(ndcg, 4) == 1.0


def test_calculate_ndcg_suboptimal_order():
    """Test NDCG@K penalizes sub-optimal document ranking."""
    # Highly relevant doc_1 is put at the end instead of first
    retrieved = ["doc_3", "doc_2", "doc_1"]
    relevance_scores = {"doc_1": 3, "doc_2": 2, "doc_3": 1}

    ndcg = RetrievalEvaluator.calculate_ndcg(retrieved, relevance_scores, k=3)
    assert ndcg < 1.0
    assert ndcg > 0.0
