import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


@patch('app.core.config.get_settings')
def test_health_endpoint(mock_settings, client):
    class MockSettings:
        service_name = "llm_connector"
        genai_api_key = "test-key"
        log_level = "INFO"
        debug = False
    
    mock_settings.return_value = MockSettings()
    
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "service" in data