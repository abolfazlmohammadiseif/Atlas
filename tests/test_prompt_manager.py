"""
Unit tests for the prompt management system.
"""

import pytest

from src.agent.prompt_manager import PromptManager


def test_prompt_manager_initialization():
    """Test that the PromptManager initializes with default templates."""
    manager = PromptManager()
    assert "researcher_system" in manager.templates
    assert "summarize_document" in manager.templates


def test_build_prompt_success():
    """Test building a prompt with valid inputs."""
    manager = PromptManager()
    prompt = manager.build_prompt(
        "summarize_document", bullet_count=5, document_text="Test document."
    )
    assert "5 key points" in prompt
    assert "Test document." in prompt


def test_build_prompt_missing_template():
    """Test that requesting an unknown template raises a ValueError."""
    manager = PromptManager()
    with pytest.raises(ValueError, match="not found"):
        manager.build_prompt("non_existent_template")


def test_build_prompt_exceeds_token_limit():
    """Test that a prompt exceeding the token limit raises a ValueError."""
    # Initialize with an artificially low token limit (e.g., 5 tokens)
    manager = PromptManager(max_prompt_tokens=5)

    with pytest.raises(ValueError, match="Prompt too long"):
        manager.build_prompt(
            "summarize_document",
            bullet_count=3,
            document_text="This is a very long document that will definitely exceed five tokens.",
        )
