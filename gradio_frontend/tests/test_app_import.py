import pytest


def test_import_main_module():
    try:
        from app import main
        assert main is not None
    except ImportError as e:
        pytest.fail(f"No se pudo importar el módulo main: {e}")


def test_import_config():
    try:
        from app.core.config import get_settings
        settings = get_settings()
        assert settings is not None
        assert settings.service_name == "gradio_frontend"
    except ImportError as e:
        pytest.fail(f"No se pudo importar config: {e}")


def test_import_clients():
    try:
        from app.services.llm_client import LLMClient
        from app.services.sklearn_client import SklearnClient
        from app.services.cnn_client import CNNClient
        
        assert LLMClient is not None
        assert SklearnClient is not None
        assert CNNClient is not None
    except ImportError as e:
        pytest.fail(f"No se pudo importar clients: {e}")


def test_import_interfaces():
    try:
        from app.ui.llm_interface import LLMInterface
        from app.ui.sklearn_interface import SklearnInterface
        from app.ui.cnn_interface import CNNInterface
        
        assert LLMInterface is not None
        assert SklearnInterface is not None
        assert CNNInterface is not None
    except ImportError as e:
        pytest.fail(f"No se pudo importar interfaces: {e}")