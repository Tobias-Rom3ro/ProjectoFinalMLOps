import os
import logging
import json


class LogFormatter(logging.Formatter):
    """Formateador JSON para logs estructurados."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "service": "sklearn_model",
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_data, ensure_ascii=False)


def configurar_logging() -> None:
    """Configura el sistema de logging estructurado."""
    nivel_debug = os.getenv("DEBUG", "false").lower() == "true"
    nivel = logging.DEBUG if nivel_debug else logging.INFO
    
    handler = logging.StreamHandler()
    handler.setFormatter(LogFormatter())
    
    logging.basicConfig(
        level=nivel,
        handlers=[handler]
    )


class Configuracion:
    """Configuración centralizada del servicio."""
    
    def __init__(self):
        self.mlflow_tracking_uri: str = os.getenv(
            "MLFLOW_TRACKING_URI",
            "http://127.0.0.1:5000/"
        )
        self.puerto_servicio: int = int(os.getenv("PORT", "8001"))
        self.ruta_modelo: str = os.getenv(
            "MODEL_PATH",
            "/app/models/wine_classifier.pkl"
        )
        self.nombre_experimento: str = os.getenv(
            "MLFLOW_EXPERIMENT_NAME",
            "sklearn_wine_classifier"
        )
        
    def validar(self) -> None:
        """Valida que la configuración sea correcta."""
        if not self.mlflow_tracking_uri:
            raise ValueError(
                "MLFLOW_TRACKING_URI no está configurado correctamente"
            )


configuracion = Configuracion()
configurar_logging()
logger = logging.getLogger(__name__)