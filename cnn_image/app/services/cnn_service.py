import logging
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
from PIL import Image
import tensorflow as tf
from tensorflow import keras

logger = logging.getLogger(__name__)


class CNNService:
    def __init__(self, cnn_model_path: str, image_size: int, num_classes: int):
        self.cnn_model_path = cnn_model_path
        self.image_size = image_size
        self.num_classes = num_classes
        self.model = None
        self.class_names = [str(i) for i in range(num_classes)]
        self._load_model()
    
    def _load_model(self) -> None:
        try:
            model_file = Path(self.cnn_model_path)
            if not model_file.exists():
                logger.warning(
                    f"Model file not found at {self.cnn_model_path}. "
                    "Model needs to be trained first."
                )
                return
            
            self.model = keras.models.load_model(self.cnn_model_path)
            logger.info(f"Model loaded successfully from {self.cnn_model_path}")
        
        except Exception as error:
            logger.error(f"Failed to load model: {error}")
            raise RuntimeError(f"Could not load CNN model: {error}")
    
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        if image.mode != 'L':
            image = image.convert('L')
        
        image = image.resize((self.image_size, self.image_size))
        
        image_array = np.array(image)
        image_array = image_array.astype('float32') / 255.0
        image_array = np.expand_dims(image_array, axis=-1)
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    
    def predict(self, image: Image.Image) -> Dict[str, any]:
        if self.model is None:
            logger.error("Model not loaded, cannot make prediction")
            raise RuntimeError(
                "Model not available. Please train the model first."
            )
        
        try:
            processed_image = self.preprocess_image(image)
            
            predictions = self.model.predict(processed_image, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])
            
            probabilities = {
                self.class_names[i]: float(predictions[0][i])
                for i in range(self.num_classes)
            }
            
            logger.info(
                f"Prediction: class={predicted_class}, "
                f"confidence={confidence:.4f}"
            )
            
            return {
                "predicted_class": int(predicted_class),
                "confidence": confidence,
                "probabilities": probabilities
            }
        
        except Exception as error:
            logger.error(f"Error during prediction: {error}")
            raise RuntimeError(f"Failed to make prediction: {error}")
    
    def is_available(self) -> bool:
        return self.model is not None
    
    def get_model_info(self) -> Dict[str, any]:
        return {
            "model_type": "CNN for MNIST digit classification",
            "input_size": f"{self.image_size}x{self.image_size} grayscale",
            "num_classes": self.num_classes,
            "classes": self.class_names,
            "description": "Classifies handwritten digits (0-9)",
            "limitations": [
                "Only works with handwritten digits",
                "Best performance on centered, single digits",
                "Grayscale images only",
                "Input should be 28x28 pixels for optimal results"
            ]
        }