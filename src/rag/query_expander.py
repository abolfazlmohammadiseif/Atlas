"""
Query Expansion module for RAG pipeline.
Expands a user's single query into multiple variations using an LLM to improve retrieval recall.
"""

from openai import OpenAI

from src.utils.logger import logger


class QueryExpander:
    """Handles query rewriting and expansion using an LLM."""

    def __init__(self, model: str = "gpt-3.5-turbo") -> None:
        """Initializes the QueryExpander.

        Args:
            model: The LLM model to use for query rewriting.
        """
        self.client = OpenAI()
        self.model = model
        logger.info("QueryExpander initialized with model: '%s'", self.model)

    def expand_query(self, query: str, num_queries: int = 3) -> list[str]:
        """Rewrites a query into multiple distinct variations.

        Args:
            query: Original user question.
            num_queries: Number of additional query variations to generate.

        Returns:
            List of queries including the original query and generated variations.
        """
        if not query.strip():
            return []

        system_prompt = (
            "You are an AI assistant specialized in information retrieval.\n"
            f"Your task is to generate {num_queries} alternative versions of the user's query.\n"
            "Each version must capture different synonyms, phrasing, or technical aspects of the same intent.\n"
            "Provide the generated queries separated strictly by newlines, with no bullet points or numbering."
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                temperature=0.4,
            )
            content = response.choices[0].message.content or ""
            generated_queries = [line.strip() for line in content.split("\n") if line.strip()]

            # Ensure original query is included and duplicates are removed while preserving order
            all_queries = [query] + [q for q in generated_queries if q != query]
            logger.info("Expanded query into %d variations.", len(all_queries))
            return all_queries
        except Exception as e:
            logger.error("Failed to expand query: %s", e)
            return [query]
