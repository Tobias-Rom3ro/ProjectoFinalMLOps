import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
import mlflow

from app.config import configuracion
from app.model import gestor_modelo
from app.schemas import (
    CaracteristicasVino,
    PrediccionRespuesta,
    EstadoSalud
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def ciclo_vida_aplicacion(app: FastAPI) -> AsyncGenerator:
    """Gestiona el ciclo de vida de la aplicación."""
    logger.info("Iniciando servicio sklearn_model")
    
    try:
        mlflow.set_tracking_uri(configuracion.mlflow_tracking_uri)
        logger.info(
            f"MLflow configurado: {configuracion.mlflow_tracking_uri}"
        )
    except Exception as error:
        logger.warning(f"No se pudo conectar a MLflow: {error}")
    
    try:
        gestor_modelo.cargar_modelo()
    except Exception as error:
        logger.error(f"Error al cargar el modelo: {error}")
    
    yield
    
    logger.info("Cerrando servicio sklearn_model")


app = FastAPI(
    title="Servicio ML Clásico - Wine Classifier",
    description="API para clasificación de vinos usando scikit-learn",
    version="1.0.0",
    lifespan=ciclo_vida_aplicacion
)


@app.get(
    "/health",
    response_model=EstadoSalud,
    tags=["Salud"]
)
async def verificar_salud() -> EstadoSalud:
    """Verifica el estado de salud del servicio."""
    mlflow_conectado = False
    
    try:
        mlflow.set_tracking_uri(configuracion.mlflow_tracking_uri)
        mlflow.get_tracking_uri()
        mlflow_conectado = True
    except Exception:
        pass
    
    return EstadoSalud(
        estado="healthy" if gestor_modelo.esta_cargado() else "degraded",
        modelo_cargado=gestor_modelo.esta_cargado(),
        mlflow_conectado=mlflow_conectado
    )


@app.post(
    "/predict",
    response_model=PrediccionRespuesta,
    tags=["Predicción"],
    status_code=status.HTTP_200_OK
)
async def predecir_vino(
    caracteristicas: CaracteristicasVino
) -> PrediccionRespuesta:
    """
    Clasifica un vino según sus características químicas.
    
    El modelo clasifica vinos en tres categorías basándose en 13 
    características químicas. Devuelve la clase predicha junto con 
    las probabilidades para cada clase.
    """
    if not gestor_modelo.esta_cargado():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="El modelo no está disponible. Ejecute el entrenamiento."
        )
    
    try:
        features_list = [
            caracteristicas.alcohol,
            caracteristicas.acido_malico,
            caracteristicas.ceniza,
            caracteristicas.alcalinidad_ceniza,
            caracteristicas.magnesio,
            caracteristicas.fenoles_totales,
            caracteristicas.flavonoides,
            caracteristicas.fenoles_no_flavonoides,
            caracteristicas.proantocianinas,
            caracteristicas.intensidad_color,
            caracteristicas.matiz,
            caracteristicas.od280_od315,
            caracteristicas.prolina
        ]
        
        clase_predicha, probabilidades = gestor_modelo.predecir(
            features_list
        )
        
        return PrediccionRespuesta(
            clase_predicha=clase_predicha,
            nombre_clase=gestor_modelo.obtener_nombre_clase(clase_predicha),
            probabilidades=probabilidades,
            confianza=max(probabilidades)
        )
        
    except Exception as error:
        logger.error(f"Error en la predicción: {error}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al realizar la predicción: {str(error)}"
        )


@app.exception_handler(Exception)
async def manejador_excepciones_global(request, exc):
    """Maneja excepciones no capturadas."""
    logger.error(f"Error no manejado: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Error interno del servidor"}
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=configuracion.puerto_servicio,
        reload=False
    )