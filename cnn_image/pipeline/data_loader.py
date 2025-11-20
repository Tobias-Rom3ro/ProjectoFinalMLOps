import numpy as np
from tensorflow import keras
from typing import Tuple
import logging


logger = logging.getLogger(__name__)


def load_mnist_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)
        
        logger.info(
            f"MNIST data loaded: {x_train.shape[0]} training samples, "
            f"{x_test.shape[0]} test samples"
        )
        
        return x_train, y_train, x_test, y_test
    
    except Exception as error:
        logger.error(f"Error loading MNIST data: {error}")
        raise