"""
Unit tests for the token counting utility.
"""

from src.utils.tokens import count_tokens

def test_count_tokens_standard_text():
    """Test counting tokens for a standard English sentence."""
    text = "Hello, world! This is a test."
    # We expect a specific token count for this exact string using cl100k_base
    expected_tokens = 9 
    assert count_tokens(text) == expected_tokens

def test_count_tokens_empty_string():
    """Test counting tokens for an empty string."""
    assert count_tokens("") == 0

def test_count_tokens_fallback_model():
    """Test that an unknown model name falls back correctly without crashing."""
    text = "Testing fallback mechanism."
    # It should use cl100k_base under the hood
    tokens_standard = count_tokens(text, model_name="gpt-4o-mini")
    tokens_fallback = count_tokens(text, model_name="unknown-model-xyz")
    
    assert tokens_standard == tokens_fallback

