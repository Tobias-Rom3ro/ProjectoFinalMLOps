import pytest
from unittest.mock import Mock, patch
from app.services.gemini_service import GeminiService


@pytest.fixture
def mock_genai_client():
    with patch('app.services.gemini_service.genai.Client') as mock:
        yield mock


def test_gemini_service_initialization(mock_genai_client):
    service = GeminiService(api_key="test-key")
    
    assert service.client is not None
    mock_genai_client.assert_called_once_with(api_key="test-key")


def test_gemini_service_generate_response(mock_genai_client):
    mock_instance = Mock()
    mock_response = Mock()
    mock_response.text = "This is a test response"
    mock_instance.models.generate_content.return_value = mock_response
    mock_genai_client.return_value = mock_instance
    
    service = GeminiService(api_key="test-key")
    response = service.generate_response(
        prompt="Test prompt",
        context=None,
        model="gemini-2.0-flash-exp"
    )
    
    assert response == "This is a test response"
    mock_instance.models.generate_content.assert_called_once()


def test_gemini_service_generate_response_with_context(mock_genai_client):
    mock_instance = Mock()
    mock_response = Mock()
    mock_response.text = "Response with context"
    mock_instance.models.generate_content.return_value = mock_response
    mock_genai_client.return_value = mock_instance
    
    service = GeminiService(api_key="test-key")
    response = service.generate_response(
        prompt="Test prompt",
        context="Test context",
        model="gemini-2.0-flash-exp"
    )
    
    assert response == "Response with context"
    call_args = mock_instance.models.generate_content.call_args
    assert "Context: Test context" in call_args[1]["contents"]


def test_gemini_service_is_available(mock_genai_client):
    service = GeminiService(api_key="test-key")
    
    assert service.is_available() is True


def test_build_prompt_without_context(mock_genai_client):
    service = GeminiService(api_key="test-key")
    
    prompt = service._build_prompt("Test prompt", None)
    
    assert prompt == "Test prompt"


def test_build_prompt_with_context(mock_genai_client):
    service = GeminiService(api_key="test-key")
    
    prompt = service._build_prompt("Test prompt", "Test context")
    
    assert "Context: Test context" in prompt
    assert "Query: Test prompt" in prompt