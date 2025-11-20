import logging
from typing import Dict, List
import httpx

logger = logging.getLogger(__name__)


class SklearnClient:
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        logger.info(f"Sklearn Client inicializado con URL: {self.base_url}")
    
    def verificar_salud(self) -> Dict[str, any]:
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(f"{self.base_url}/health")
                response.raise_for_status()
                return response.json()
        except Exception as error:
            logger.error(f"Error al verificar salud del modelo: {error}")
            raise
    
    def predecir(
        self,
        alcohol: float,
        acido_malico: float,
        ceniza: float,
        alcalinidad_ceniza: float,
        magnesio: float,
        fenoles_totales: float,
        flavonoides: float,
        fenoles_no_flavonoides: float,
        proantocianinas: float,
        intensidad_color: float,
        matiz: float,
        od280_od315: float,
        prolina: float
    ) -> Dict[str, any]:
        try:
            payload = {
                "alcohol": alcohol,
                "acido_malico": acido_malico,
                "ceniza": ceniza,
                "alcalinidad_ceniza": alcalinidad_ceniza,
                "magnesio": magnesio,
                "fenoles_totales": fenoles_totales,
                "flavonoides": flavonoides,
                "fenoles_no_flavonoides": fenoles_no_flavonoides,
                "proantocianinas": proantocianinas,
                "intensidad_color": intensidad_color,
                "matiz": matiz,
                "od280_od315": od280_od315,
                "prolina": prolina
            }
            
            logger.info("Enviando datos para predicción al modelo sklearn")
            
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.base_url}/predict",
                    json=payload
                )
                response.raise_for_status()
                data = response.json()
                
                logger.info(
                    f"Predicción recibida: clase {data.get('clase_predicha')}"
                )
                return data
        
        except httpx.TimeoutException:
            error_msg = "Tiempo de espera agotado"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        except httpx.HTTPStatusError as error:
            error_msg = f"Error HTTP {error.response.status_code}"
            logger.error(f"{error_msg}: {error}")
            raise RuntimeError(error_msg)
        
        except Exception as error:
            error_msg = f"Error al realizar predicción: {str(error)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)