# file: llm_connector/tests/conftest.py
import sys
import os
from pathlib import Path

# Añadir el directorio padre al path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

# Configurar variables de entorno para tests
os.environ["GENAI_API_KEY"] = "test-api-key-12345"
os.environ["SERVICE_NAME"] = "llm_connector"
os.environ["LOG_LEVEL"] = "ERROR"
os.environ["DEBUG"] = "False"
