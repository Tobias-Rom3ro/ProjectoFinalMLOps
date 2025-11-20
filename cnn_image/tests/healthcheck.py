import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock

from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_cnn_service():
    with patch('app.api.routes.get_cnn_service') as mock:
        service = Mock()
        service.is_available.return_value = True
        mock.return_value.__enter__ = Mock(return_value=service)
        mock.return_value.__exit__ = Mock(return_value=False)
        yield service


def test_health_endpoint(client, mock_cnn_service):
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "service" in data
    assert "model_loaded" in data