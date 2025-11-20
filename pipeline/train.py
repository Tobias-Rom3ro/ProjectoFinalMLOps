import logging
import pickle
from pathlib import Path
from datetime import datetime

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from app.config import configuracion
from pipeline.utils import (
    cargar_datos_wine,
    dividir_datos,
    crear_escalador
)

logger = logging.getLogger(__name__)


def entrenar_modelo() -> None:
    """Entrena el modelo de clasificación de vinos y registra en MLflow."""
    logger.info("Iniciando entrenamiento del modelo")
    
    mlflow.set_tracking_uri(configuracion.mlflow_tracking_uri)
    mlflow.set_experiment(configuracion.nombre_experimento)
    
    X, y, nombres_features, nombres_clases = cargar_datos_wine()
    
    X_train, X_test, y_train, y_test = dividir_datos(X, y)
    
    parametros = {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42
    }
    
    with mlflow.start_run(
        run_name=f"wine_classifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ):
        logger.info("Iniciando run de MLflow")
        
        mlflow.log_params(parametros)
        mlflow.log_param("dataset", "wine")
        mlflow.log_param("n_samples_train", X_train.shape[0])
        mlflow.log_param("n_samples_test", X_test.shape[0])
        mlflow.log_param("n_features", X_train.shape[1])
        
        pipeline = Pipeline([
            ("escalador", crear_escalador()),
            ("clasificador", RandomForestClassifier(**parametros))
        ])
        
        logger.info("Entrenando pipeline...")
        pipeline.fit(X_train, y_train)
        
        y_pred_train = pipeline.predict(X_train)
        y_pred_test = pipeline.predict(X_test)
        
        metricas_train = {
            "train_accuracy": accuracy_score(y_train, y_pred_train),
            "train_precision": precision_score(
                y_train,
                y_pred_train,
                average="weighted"
            ),
            "train_recall": recall_score(
                y_train,
                y_pred_train,
                average="weighted"
            ),
            "train_f1": f1_score(y_train, y_pred_train, average="weighted")
        }
        
        metricas_test = {
            "test_accuracy": accuracy_score(y_test, y_pred_test),
            "test_precision": precision_score(
                y_test,
                y_pred_test,
                average="weighted"
            ),
            "test_recall": recall_score(
                y_test,
                y_pred_test,
                average="weighted"
            ),
            "test_f1": f1_score(y_test, y_pred_test, average="weighted")
        }
        
        mlflow.log_metrics(metricas_train)
        mlflow.log_metrics(metricas_test)
        
        logger.info("Métricas de entrenamiento:")
        for metrica, valor in metricas_train.items():
            logger.info(f"  {metrica}: {valor:.4f}")
            
        logger.info("Métricas de prueba:")
        for metrica, valor in metricas_test.items():
            logger.info(f"  {metrica}: {valor:.4f}")
        
        reporte = classification_report(
            y_test,
            y_pred_test,
            target_names=nombres_clases
        )
        logger.info(f"\nReporte de clasificación:\n{reporte}")
        
        matriz_confusion = confusion_matrix(y_test, y_pred_test)
        logger.info(f"\nMatriz de confusión:\n{matriz_confusion}")
        
        ruta_modelo = Path(configuracion.ruta_modelo)
        ruta_modelo.parent.mkdir(parents=True, exist_ok=True)
        
        with open(ruta_modelo, 'wb') as archivo:
            pickle.dump(pipeline, archivo)
        
        logger.info(f"Modelo guardado en {ruta_modelo}")
        
        mlflow.sklearn.log_model(
            pipeline,
            "model",
            registered_model_name="wine_classifier"
        )
        
        mlflow.log_artifact(str(ruta_modelo))
        
        Path("/tmp").mkdir(parents=True, exist_ok=True)
        with open("/tmp/classification_report.txt", "w") as archivo:
            archivo.write(reporte)
        mlflow.log_artifact("/tmp/classification_report.txt")
        
        logger.info("Entrenamiento completado y registrado en MLflow")


if __name__ == "__main__":
    from app.config import configurar_logging
    
    configurar_logging()
    entrenar_modelo()