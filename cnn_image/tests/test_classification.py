# file: cnn_image/tests/test_classification.py
import pytest
import io
from PIL import Image
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock, MagicMock

from app.main import app


@pytest.fixture
def mock_dependencies():
    cnn_service = Mock()
    cnn_service.is_available.return_value = True
    cnn_service.predict.return_value = {
        "predicted_class": 5,
        "confidence": 0.95,
        "probabilities": {str(i): 0.1 for i in range(10)}
    }
    
    filter_service = Mock()
    filter_service.apply_filter.return_value = Image.new('L', (28, 28))
    
    with patch('app.api.dependencies._cnn_service', new=cnn_service), \
         patch('app.api.dependencies._filter_service', new=filter_service):
        
        yield cnn_service, filter_service


@pytest.fixture
def client(mock_dependencies):
    return TestClient(app)


def create_test_image():
    img = Image.new('L', (28, 28), color=128)
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return img_bytes


def test_classify_endpoint(client, mock_dependencies):
    test_image = create_test_image()
    
    response = client.post(
        "/classify",
        files={"file": ("test.png", test_image, "image/png")},
        data={"filter_name": "none"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "predicted_class" in data
    assert "confidence" in data
    assert "probabilities" in data
    assert "filter_applied" in data


def test_classify_with_filter(client, mock_dependencies):
    test_image = create_test_image()
    
    response = client.post(
        "/classify",
        files={"file": ("test.png", test_image, "image/png")},
        data={"filter_name": "blur"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["filter_applied"] == "blur"