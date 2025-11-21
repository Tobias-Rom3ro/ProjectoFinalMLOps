import logging
from typing import Tuple
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def cargar_datos_wine() -> Tuple[np.ndarray, np.ndarray, list, list]:
    """
    Carga el dataset Wine de scikit-learn.
    
    Returns:
        Tupla con (X, y, nombres_features, nombres_clases)
    """
    logger.info("Cargando dataset Wine")
    
    datos = load_wine()
    X = datos.data
    y = datos.target
    nombres_features = datos.feature_names
    nombres_clases = datos.target_names.tolist()
    
    logger.info(
        f"Dataset cargado: {X.shape[0]} muestras, "
        f"{X.shape[1]} características, {len(nombres_clases)} clases"
    )
    
    return X, y, nombres_features, nombres_clases


def dividir_datos(
    X: np.ndarray,
    y: np.ndarray,
    tamanio_test: float = 0.2,
    semilla: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Divide los datos en conjuntos de entrenamiento y prueba.
    
    Args:
        X: Matriz de características
        y: Vector de etiquetas
        tamanio_test: Proporción del conjunto de prueba
        semilla: Semilla para reproducibilidad
        
    Returns:
        Tupla con (X_train, X_test, y_train, y_test)
    """
    logger.info(
        f"Dividiendo datos: {(1-tamanio_test)*100:.0f}% entrenamiento, "
        f"{tamanio_test*100:.0f}% prueba"
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=tamanio_test,
        random_state=semilla,
        stratify=y
    )
    
    logger.info(
        f"Conjunto de entrenamiento: {X_train.shape[0]} muestras"
    )
    logger.info(f"Conjunto de prueba: {X_test.shape[0]} muestras")
    
    return X_train, X_test, y_train, y_test


def crear_escalador() -> StandardScaler:
    """
    Crea un escalador estándar para normalizar características.
    
    Returns:
        Objeto StandardScaler
    """
    return StandardScaler()