"""
Structured Output Extraction Engine.
Enforces Pydantic schemas on LLM outputs using schema injection and auto-correction retries.
"""

import json
from typing import TypeVar

from pydantic import BaseModel, Field, ValidationError

from src.agent.llm_client import LLMClient
from src.utils.logger import logger

# Generic Type for any Pydantic Model
T = TypeVar("T", bound=BaseModel)


# --- 1. Complex Schema Definitions for Real-World Research Analysis ---


class ResearchFact(BaseModel):
    """Represents an individual fact extracted from research context."""

    fact: str = Field(description="The core factual claim extracted.")
    confidence_score: float = Field(
        ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0"
    )
    category: str = Field(description="Category of the fact (e.g., Technical, Financial, Security)")


class ResearchReport(BaseModel):
    """Complete structured response schema for document analysis."""

    title: str = Field(description="Concise summary title of the research report.")
    summary: str = Field(description="Executive summary of the document.")
    extracted_facts: list[ResearchFact] = Field(
        description="List of structured facts found in the context."
    )
    risk_level: str = Field(
        default="LOW", description="Assessed risk level: LOW, MEDIUM, HIGH, or CRITICAL"
    )


# --- 2. Robust Extractor with Retry Loop ---


class StructuredOutputExtractor:
    """Enforces structured JSON generation from LLM using Pydantic models."""

    def __init__(self, llm_client: LLMClient) -> None:
        """Initializes extractor with an LLM Client instance.

        Args:
            llm_client: Configured LLMClient instance.
        """
        self.client = llm_client

    def extract(
        self,
        schema_class: type[T],
        raw_text: str,
        max_retries: int = 3,
    ) -> T:
        """Extracts structured data from raw text according to a Pydantic schema.

        Includes an automatic feedback loop to retry if validation fails.

        Args:
            schema_class: The Pydantic model class defining desired output structure.
            raw_text: Source document or context to extract from.
            max_retries: Maximum correction attempts upon JSON/Validation errors.

        Returns:
            T: Instantiated Pydantic model with validated data.

        Raises:
            ValidationError: If LLM fails to satisfy schema after max retries.
        """
        # Generate JSON Schema string from Pydantic Model
        json_schema = json.dumps(schema_class.model_json_schema(), indent=2)

        system_prompt = (
            "You are a strict data extraction engine.\n"
            "Your task is to parse context and extract data strictly matching this JSON Schema:\n"
            f"```json\n{json_schema}\n```\n"
            "CRITICAL RULES:\n"
            "1. Output ONLY valid raw JSON matching the schema.\n"
            "2. Do NOT wrap output in markdown code blocks or add explanatory text.\n"
            "3. Ensure all fields and types match the schema strictly."
        )

        current_prompt = f"Extract structured information from this context:\n\n{raw_text}"

        for attempt in range(1, max_retries + 1):
            logger.info(
                "Extraction attempt %d/%d for schema %s",
                attempt,
                max_retries,
                schema_class.__name__,
            )

            # Send API Call
            response_text = self.client.generate_response(
                prompt=current_prompt,
                system_prompt=system_prompt,
                temperature=0.1,  # Low temperature for deterministic output
            )

            # Sanitize response (remove markdown fences if LLM accidentally added them)
            cleaned_response = response_text.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = (
                    cleaned_response.removeprefix("```json").removesuffix("```").strip()
                )
            elif cleaned_response.startswith("```"):
                cleaned_response = cleaned_response.removeprefix("```").removesuffix("```").strip()

            # Attempt JSON Parsing & Pydantic Validation
            try:
                raw_json = json.loads(cleaned_response)
                validated_data = schema_class.model_validate(raw_json)
                logger.info(
                    "Successfully validated output against %s on attempt %d.",
                    schema_class.__name__,
                    attempt,
                )
                return validated_data

            except (json.JSONDecodeError, ValidationError) as err:
                logger.warning("Attempt %d failed validation with error: %s", attempt, err)

                if attempt == max_retries:
                    logger.error("Reached maximum retries. Structured extraction failed.")
                    raise err

                # Feedback loop: Feed the error back to the LLM to fix its own mistake
                current_prompt = (
                    "Your previous JSON output was invalid and threw the following error:\n"
                    f"{err}\n\n"
                    f"Previous Invalid Output:\n{cleaned_response}\n\n"
                    "Please fix the error and provide the correct JSON strictly for the original "
                    f"context:\n{raw_text}"
                )


if __name__ == "__main__":
    # Internal Unit Test for Module Logic
    print("Testing Pydantic Schema Generation:")
    print(json.dumps(ResearchReport.model_json_schema(), indent=2))
