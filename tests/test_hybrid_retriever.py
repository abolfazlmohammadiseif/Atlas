"""
Unit tests for the Hybrid Retriever (BM25 + RRF).
"""

from src.rag.hybrid_retriever import HybridSearcher


def test_bm25_search():
    """Test if BM25 correctly finds exact keyword matches."""
    corpus = [
        {"id": "1", "text": "The quick brown fox jumps over the lazy dog."},
        {"id": "2", "text": "FastAPI is a modern web framework for Python."},
        {"id": "3", "text": "Machine learning models require good data."},
    ]

    searcher = HybridSearcher(corpus_chunks=corpus)
    results = searcher.search_bm25("FastAPI Python", top_k=2)

    assert len(results) > 0
    assert results[0]["id"] == "2"


def test_reciprocal_rank_fusion():
    """Test if RRF correctly merges and prioritizes overlapping results."""
    vector_results = [
        {"id": "doc_A", "text": "A"},  # Rank 0
        {"id": "doc_B", "text": "B"},  # Rank 1
        {"id": "doc_C", "text": "C"},  # Rank 2
    ]

    bm25_results = [
        {"id": "doc_C", "text": "C"},  # Rank 0
        {"id": "doc_A", "text": "A"},  # Rank 1
        {"id": "doc_D", "text": "D"},  # Rank 2
    ]

    # doc_A gets: 1/(60+0) + 1/(60+1) = 0.0166 + 0.0163 = 0.0329
    # doc_C gets: 1/(60+2) + 1/(60+0) = 0.0161 + 0.0166 = 0.0327
    # doc_B gets: 1/(60+1) = 0.0163
    # doc_D gets: 1/(60+2) = 0.0161
    # Expected order: A, C, B, D

    fused_results = HybridSearcher.reciprocal_rank_fusion(vector_results, bm25_results, k=60)

    assert len(fused_results) == 4
    assert fused_results[0]["id"] == "doc_A"
    assert fused_results[1]["id"] == "doc_C"
    assert fused_results[2]["id"] == "doc_B"
    assert fused_results[3]["id"] == "doc_D"
