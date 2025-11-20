import logging
from tensorflow import keras
from tensorflow.keras import layers

logger = logging.getLogger(__name__)


def build_mnist_cnn(
    input_shape: tuple = (28, 28, 1),
    num_classes: int = 10
) -> keras.Model:
    logger.info(f"Building CNN model with input shape: {input_shape}")
    
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        layers.Flatten(),
        
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    logger.info("Model architecture created")
    return model