"""
Prompt management system.
Isolates prompt templates from business logic and enforces context window limits.
"""

from typing import Any

from src.utils.logger import logger
from src.utils.tokens import count_tokens


class PromptManager:
    """Manages system and user prompt templates with token validation."""

    def __init__(self, max_prompt_tokens: int = 6000) -> None:
        """Initializes the prompt manager.

        Args:
            max_prompt_tokens: Safety threshold to prevent ContextLengthExceeded errors.
        """
        self.max_prompt_tokens = max_prompt_tokens

        # Centralized prompt repository (Can be moved to YAML later)
        self.templates = {
            "researcher_system": (
                "You are an expert Autonomous Research Agent.\n"
                "Your goal is to analyze documents and provide factual, structured insights.\n"
                "Never fabricate information."
            ),
            "summarize_document": (
                "Please summarize the following document into {bullet_count} key points.\n\n"
                "Document Context:\n{document_text}"
            ),
        }

    def build_prompt(self, template_name: str, **kwargs: Any) -> str:
        """Hydrates a template with variables and checks token limits.

        Args:
            template_name: The key of the prompt in self.templates.
            **kwargs: Dynamic variables to inject into the template.

        Returns:
            str: The fully formatted prompt.

        Raises:
            ValueError: If template does not exist or exceeds token limits.
        """
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found.")

        # Format the prompt with provided keyword arguments
        prompt = self.templates[template_name].format(**kwargs)

        # Validate token count before sending to LLM
        token_count = count_tokens(prompt)
        if token_count > self.max_prompt_tokens:
            logger.error(
                "Prompt '%s' exceeded token limit: %d > %d",
                template_name,
                token_count,
                self.max_prompt_tokens,
            )
            # In a real pipeline, we would apply a chunking or truncation strategy here
            raise ValueError(f"Prompt too long! ({token_count} tokens)")

        logger.debug("Successfully built prompt '%s' (%d tokens).", template_name, token_count)
        return prompt


if __name__ == "__main__":
    manager = PromptManager()

    # Test building a prompt
    system_prompt = manager.build_prompt("researcher_system")
    user_prompt = manager.build_prompt(
        "summarize_document",
        bullet_count=3,
        document_text="The AI agent successfully processed \
the large dataset using RAG architecture.",
    )

    print("--- System Prompt ---")
    print(system_prompt)
    print("\n--- User Prompt ---")
    print(user_prompt)
