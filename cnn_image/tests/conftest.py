import sys
import os
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

os.environ["SERVICE_NAME"] = "cnn_image"
os.environ["LOG_LEVEL"] = "ERROR"
os.environ["DEBUG"] = "False"
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
os.environ["CNN_MODEL_PATH"] = "models/mnist_cnn_model.keras"
os.environ["IMAGE_SIZE"] = "28"
os.environ["NUM_CLASSES"] = "10"