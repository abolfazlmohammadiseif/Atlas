"""
LLM Client module for abstracting interaction with Large Language Model APIs.
Supports structured response generation and error handling.
"""

from openai import APIError, OpenAI

from src.utils.config import settings
from src.utils.logger import logger


class LLMClient:
    """Wrapper class for interacting with OpenAI-compatible LLM providers."""

    def __init__(self, model_name: str = "gpt-4o-mini") -> None:
        """Initializes the LLM Client with configuration settings.

        Args:
            model_name: The target LLM model identifier.
        """
        self.model_name = model_name
        self.api_key = settings.openai_api_key

        if not self.api_key:
            logger.warning(
                "OpenAI API Key is not set in environment variables! "
                "Calls will fail unless a mock/key is provided."
            )

        # Initialize official client
        self.client = OpenAI(api_key=self.api_key or "mock-key")

    def generate_response(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful AI research assistant.",
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> str:
        """Sends a completion request to the LLM API.

        Args:
            prompt: User input query or instructions.
            system_prompt: System-level instructions defining LLM behavior.
            temperature: Sampling temperature for randomness control.
            max_tokens: Maximum tokens to generate.

        Returns:
            str: Generated text response from the model.
        """
        try:
            logger.info("Sending request to model: %s", self.model_name)

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            content = response.choices[0].message.content
            logger.info("Successfully received response from LLM.")
            return content or ""

        except APIError as err:
            logger.error("OpenAI API call failed with error: %s", err)
            raise RuntimeError(f"LLM Provider Error: {err}") from err


if __name__ == "__main__":
    # Test client instantiation without making real paid API calls
    client = LLMClient(model_name="gpt-4o-mini")
    print(f"LLM Client initialized successfully for model: {client.model_name}")
