"""
Script para optimizar hiperparámetros del modelo.
"""
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import mlflow

from app.config import configuracion, configurar_logging
from pipeline.utils import dividir_datos, crear_escalador

import logging
logger = logging.getLogger(__name__)


def optimizar_hiperparametros():
    """Busca los mejores hiperparámetros para el modelo."""
    configurar_logging()
    logger.info("Iniciando optimización de hiperparámetros")
    
    mlflow.set_tracking_uri(configuracion.mlflow_tracking_uri)
    mlflow.set_experiment(f"{configuracion.nombre_experimento}_optimization")
    
    datos = load_wine()
    X, y = datos.data, datos.target
    
    X_train, X_test, y_train, y_test = dividir_datos(X, y)
    
    pipeline = Pipeline([
        ("escalador", crear_escalador()),
        ("clasificador", RandomForestClassifier(random_state=42))
    ])
    
    param_grid = {
        'clasificador__n_estimators': [50, 100, 200],
        'clasificador__max_depth': [5, 10, 15, None],
        'clasificador__min_samples_split': [2, 5, 10],
        'clasificador__min_samples_leaf': [1, 2, 4]
    }
    
    print("\nIniciando búsqueda de hiperparámetros...")
    print(f"Combinaciones a probar: {len(param_grid['clasificador__n_estimators']) * len(param_grid['clasificador__max_depth']) * len(param_grid['clasificador__min_samples_split']) * len(param_grid['clasificador__min_samples_leaf'])}")
    
    with mlflow.start_run(run_name="hyperparameter_optimization"):
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=2
        )
        
        grid_search.fit(X_train, y_train)
        
        print("\n" + "="*70)
        print("RESULTADOS DE OPTIMIZACIÓN")
        print("="*70)
        print("\nMejores parámetros encontrados:")
        for param, valor in grid_search.best_params_.items():
            print(f"  {param}: {valor}")
        
        print(f"\nMejor score (CV): {grid_search.best_score_:.4f}")
        print(f"Score en test: {grid_search.score(X_test, y_test):.4f}")
        
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("best_cv_score", grid_search.best_score_)
        mlflow.log_metric("test_score", grid_search.score(X_test, y_test))
        
        print("\n✓ Parámetros óptimos registrados en MLflow")
        print("="*70 + "\n")


if __name__ == "__main__":
    optimizar_hiperparametros()