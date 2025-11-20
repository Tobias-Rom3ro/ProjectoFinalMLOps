# file: cnn_image/pipeline/train.py
import os
import logging
from pathlib import Path
import numpy as np
import mlflow
import mlflow.tensorflow
from tensorflow import keras
from tensorflow.keras.datasets import mnist

from pipeline.model_builder import build_mnist_cnn


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_and_preprocess_data():
    logger.info("Loading MNIST dataset")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    
    logger.info(
        f"Data loaded - Train: {x_train.shape}, Test: {x_test.shape}"
    )
    
    return (x_train, y_train), (x_test, y_test)


def train_model(
    epochs: int = 5,
    batch_size: int = 128,
    cnn_model_path: str = "models/mnist_cnn_model.keras"
):
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    
    experiment_name = "cnn_mnist_classification"
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
    
    mlflow.set_experiment(experiment_name)
    
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    with mlflow.start_run():
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("input_shape", (28, 28, 1))
        mlflow.log_param("num_classes", 10)
        mlflow.log_param("optimizer", "adam")
        mlflow.log_param("loss", "sparse_categorical_crossentropy")
        
        model = build_mnist_cnn(input_shape=(28, 28, 1), num_classes=10)
        
        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        
        logger.info("Starting model training")
        
        history = model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.1,
            verbose=1
        )
        
        for epoch in range(epochs):
            mlflow.log_metric(
                "train_loss",
                history.history['loss'][epoch],
                step=epoch
            )
            mlflow.log_metric(
                "train_accuracy",
                history.history['accuracy'][epoch],
                step=epoch
            )
            mlflow.log_metric(
                "val_loss",
                history.history['val_loss'][epoch],
                step=epoch
            )
            mlflow.log_metric(
                "val_accuracy",
                history.history['val_accuracy'][epoch],
                step=epoch
            )
        
        logger.info("Evaluating model on test set")
        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
        
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_accuracy)
        
        logger.info(f"Test accuracy: {test_accuracy:.4f}")
        
        save_path = Path(cnn_model_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        model.save(str(save_path))
        logger.info(f"Model saved to {save_path}")
        
        mlflow.tensorflow.log_model(model, "model")
        mlflow.log_artifact(str(save_path))
        
        logger.info("Training completed successfully")
        
        return model, history


if __name__ == "__main__":
    train_model(epochs=5, batch_size=128)