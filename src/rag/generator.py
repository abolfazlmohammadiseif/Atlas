"""
Generator module for the RAG pipeline.
Takes the user query and retrieved context, and generates an answer using an LLM.
"""

from openai import OpenAI

from src.utils.logger import logger


class RAGGenerator:
    """Handles text generation using an LLM based on provided context."""

    def __init__(self, model: str = "gpt-3.5-turbo") -> None:
        """Initializes the Generator with an OpenAI client.

        Args:
            model: The name of the LLM model to use (default: gpt-3.5-turbo).
        """
        self.client = OpenAI()
        self.model = model
        logger.info("RAG Generator initialized with model: '%s'", self.model)

    def generate_answer(self, query: str, context: str) -> str:
        """Generates an answer for the user query strictly based on the context.

        Args:
            query: The user's question.
            context: The retrieved text chunks formatted as a single string.

        Returns:
            The generated response from the LLM.
        """
        system_prompt = (
            "You are a highly capable and accurate AI assistant.\n"
            "Your task is to answer the user's question based strictly on the provided context.\n"
            "If the answer cannot be found in the context, do not guess or use outside knowledge. "
            "Simply reply with: 'I cannot answer this based on the provided documents.'\n\n"
            "--- CONTEXT ---\n"
            f"{context}\n"
            "---------------"
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            temperature=0.0,  # Zero temperature for maximum determinism and minimum hallucination
        )

        answer = response.choices[0].message.content
        logger.info("Successfully generated answer for the query.")
        return answer or ""
