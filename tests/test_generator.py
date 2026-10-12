"""
Unit tests for the RAG Generator module.
"""

from unittest.mock import MagicMock, patch

from src.rag.generator import RAGGenerator


@patch("src.rag.generator.OpenAI")
def test_generate_answer(mock_openai_class):
    """Test that the generator correctly constructs the prompt and calls the LLM."""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client

    # Mock the response structure from OpenAI chat completions
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "This is a mocked answer based on context."
    mock_client.chat.completions.create.return_value = mock_response

    generator = RAGGenerator()

    query = "What is the capital of France?"
    context = "Paris is the capital of France."

    answer = generator.generate_answer(query=query, context=context)

    assert answer == "This is a mocked answer based on context."
    mock_client.chat.completions.create.assert_called_once()

    # Verify the parameters passed to the API
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "gpt-3.5-turbo"
    assert call_kwargs["temperature"] == 0.0
    assert len(call_kwargs["messages"]) == 2
    assert call_kwargs["messages"][1]["content"] == "What is the capital of France?"
    assert "Paris is the capital of France." in call_kwargs["messages"][0]["content"]
