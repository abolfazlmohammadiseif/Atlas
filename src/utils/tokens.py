"""
Tokenization utility module.
Provides fast token counting for LLM context window management.
"""

import tiktoken

from src.utils.logger import logger


def count_tokens(text: str, model_name: str = "gpt-4o-mini") -> int:
    """Calculates the exact number of tokens for a given text and model.

    Args:
        text: The string to tokenize.
        model_name: The LLM model name to fetch the correct encoding.

    Returns:
        int: Number of tokens.
    """
    try:
        # Get the specific encoding for the requested model
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        # Fallback for unknown models (uses standard GPT-4 encoding)
        logger.debug("Model %s not found in tiktoken. Falling back to cl100k_base.", model_name)
        encoding = tiktoken.get_encoding("cl100k_base")

    token_count = len(encoding.encode(text))
    return token_count


if __name__ == "__main__":
    sample_text = "AI Engineering is about building reliable, scalable systems."
    tokens = count_tokens(sample_text)
    print(f"Text: '{sample_text}'")
    print(f"Token Count: {tokens}")
