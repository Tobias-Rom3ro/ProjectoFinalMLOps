import sys
import os
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

os.environ["SERVICE_NAME"] = "gradio_frontend"
os.environ["LOG_LEVEL"] = "ERROR"
os.environ["DEBUG"] = "False"
os.environ["LLM_CONNECTOR_URL"] = "http://localhost:8000"
os.environ["SKLEARN_MODEL_URL"] = "http://localhost:8001"
os.environ["CNN_IMAGE_URL"] = "http://localhost:8002"
os.environ["REQUEST_TIMEOUT"] = "30"