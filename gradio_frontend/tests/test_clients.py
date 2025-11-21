import pytest
from unittest.mock import Mock, patch
from app.services.llm_client import LLMClient
from app.services.sklearn_client import SklearnClient
from app.services.cnn_client import CNNClient


@pytest.fixture
def mock_httpx_client():
    with patch('httpx.Client') as mock:
        yield mock


def test_llm_client_initialization():
    client = LLMClient(base_url="http://localhost:8000", timeout=30)
    
    assert client.base_url == "http://localhost:8000"
    assert client.timeout == 30


def test_llm_client_consultar_success(mock_httpx_client):
    mock_response = Mock()
    mock_response.json.return_value = {
        "response": "Esta es una respuesta de prueba"
    }
    mock_response.raise_for_status = Mock()
    
    mock_context = Mock()
    mock_context.__enter__ = Mock(return_value=mock_context)
    mock_context.__exit__ = Mock(return_value=False)
    mock_context.post = Mock(return_value=mock_response)
    
    mock_httpx_client.return_value = mock_context
    
    client = LLMClient(base_url="http://localhost:8000")
    resultado = client.consultar("Pregunta de prueba")
    
    assert resultado == "Esta es una respuesta de prueba"


def test_sklearn_client_initialization():
    client = SklearnClient(base_url="http://localhost:8001", timeout=30)
    
    assert client.base_url == "http://localhost:8001"
    assert client.timeout == 30


def test_sklearn_client_predecir_success(mock_httpx_client):
    mock_response = Mock()
    mock_response.json.return_value = {
        "clase_predicha": 1,
        "nombre_clase": "Clase 1",
        "confianza": 0.95,
        "probabilidades": [0.02, 0.95, 0.03]
    }
    mock_response.raise_for_status = Mock()
    
    mock_context = Mock()
    mock_context.__enter__ = Mock(return_value=mock_context)
    mock_context.__exit__ = Mock(return_value=False)
    mock_context.post = Mock(return_value=mock_response)
    
    mock_httpx_client.return_value = mock_context
    
    client = SklearnClient(base_url="http://localhost:8001")
    resultado = client.predecir(
        alcohol=13.0,
        acido_malico=2.0,
        ceniza=2.4,
        alcalinidad_ceniza=20.0,
        magnesio=100.0,
        fenoles_totales=2.5,
        flavonoides=2.0,
        fenoles_no_flavonoides=0.3,
        proantocianinas=1.5,
        intensidad_color=5.0,
        matiz=1.0,
        od280_od315=3.0,
        prolina=1000.0
    )
    
    assert resultado["clase_predicha"] == 1
    assert resultado["confianza"] == 0.95


def test_cnn_client_initialization():
    client = CNNClient(base_url="http://localhost:8002", timeout=30)
    
    assert client.base_url == "http://localhost:8002"
    assert client.timeout == 30


def test_cnn_client_verificar_salud_success(mock_httpx_client):
    mock_response = Mock()
    mock_response.json.return_value = {
        "status": "healthy",
        "service": "cnn_image",
        "model_loaded": True
    }
    mock_response.raise_for_status = Mock()
    
    mock_context = Mock()
    mock_context.__enter__ = Mock(return_value=mock_context)
    mock_context.__exit__ = Mock(return_value=False)
    mock_context.get = Mock(return_value=mock_response)
    
    mock_httpx_client.return_value = mock_context
    
    client = CNNClient(base_url="http://localhost:8002")
    salud = client.verificar_salud()
    
    assert salud["status"] == "healthy"
    assert salud["model_loaded"] is True


def test_cnn_client_obtener_info_modelo_success(mock_httpx_client):
    mock_response = Mock()
    mock_response.json.return_value = {
        "model_type": "CNN for MNIST",
        "num_classes": 10,
        "available_filters": ["none", "blur", "edge_detection", "sharpen"]
    }
    mock_response.raise_for_status = Mock()
    
    mock_context = Mock()
    mock_context.__enter__ = Mock(return_value=mock_context)
    mock_context.__exit__ = Mock(return_value=False)
    mock_context.get = Mock(return_value=mock_response)
    
    mock_httpx_client.return_value = mock_context
    
    client = CNNClient(base_url="http://localhost:8002")
    info = client.obtener_info_modelo()
    
    assert info["model_type"] == "CNN for MNIST"
    assert info["num_classes"] == 10
    assert len(info["available_filters"]) == 4