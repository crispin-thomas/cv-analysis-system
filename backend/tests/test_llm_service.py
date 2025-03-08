import pytest
from unittest.mock import MagicMock, patch
from app.services.llm_service import LLMService


class TestConfig:
    LLM_PROVIDER = "anthropic"
    ANTHROPIC_API_KEY = "test_key"
    ANTHROPIC_MODEL = "claude-3-opus-20240229"
    OPENAI_API_KEY = "test_key"
    OPENAI_MODEL = "gpt-4o"


@pytest.fixture
def anthropic_config():
    config = TestConfig()
    config.LLM_PROVIDER = "anthropic"
    return config


@pytest.fixture
def openai_config():
    config = TestConfig()
    config.LLM_PROVIDER = "openai"
    return config


@patch("anthropic.Anthropic")
def test_anthropic_setup(mock_anthropic, anthropic_config):
    """Test that Anthropic client is properly set up"""
    llm_service = LLMService(anthropic_config)
    mock_anthropic.assert_called_once_with(api_key="test_key")
    assert llm_service.model == "claude-3-opus-20240229"


@patch("openai.OpenAI")
def test_openai_setup(mock_openai, openai_config):
    """Test that OpenAI client is properly set up"""
    llm_service = LLMService(openai_config)
    mock_openai.assert_called_once_with(api_key="test_key")
    assert llm_service.model == "gpt-4o"


@patch("anthropic.Anthropic")
def test_anthropic_query(mock_anthropic, anthropic_config):
    """Test querying the Anthropic API"""
    # Set up mock
    mock_client = MagicMock()
    mock_message = MagicMock()
    mock_content = MagicMock()
    mock_content.text = "Test response"
    mock_message.content = [mock_content]
    mock_client.messages.create.return_value = mock_message
    mock_anthropic.return_value = mock_client

    # Create service and query
    llm_service = LLMService(anthropic_config)
    result = llm_service.query("Test prompt")

    # Verify result
    assert result == "Test response"
    mock_client.messages.create.assert_called_once_with(
        model="claude-3-opus-20240229",
        temperature=0.1,
        max_tokens=4000,
        messages=[{"role": "user", "content": "Test prompt"}],
    )


@patch("openai.OpenAI")
def test_openai_query(mock_openai, openai_config):
    """Test querying the OpenAI API"""
    # Set up mock
    mock_client = MagicMock()
    mock_chat = MagicMock()
    mock_completion = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.content = "Test response"
    mock_choice.message = mock_message
    mock_completion.choices = [mock_choice]
    mock_chat.completions.create.return_value = mock_completion
    mock_client.chat = mock_chat
    mock_openai.return_value = mock_client

    # Create service and query
    llm_service = LLMService(openai_config)
    result = llm_service.query("Test prompt")

    # Verify result
    assert result == "Test response"
    mock_chat.completions.create.assert_called_once()
