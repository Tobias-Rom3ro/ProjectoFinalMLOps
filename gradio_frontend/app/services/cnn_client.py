import logging
from typing import Dict, Optional
import httpx
from pathlib import Path

logger = logging.getLogger(__name__)


class CNNClient:
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        logger.info(f"CNN Client inicializado con URL: {self.base_url}")
    
    def verificar_salud(self) -> Dict[str, any]:
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(f"{self.base_url}/health")
                response.raise_for_status()
                return response.json()
        except Exception as error:
            logger.error(f"Error al verificar salud de CNN: {error}")
            raise
    
    def obtener_info_modelo(self) -> Dict[str, any]:
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(f"{self.base_url}/model/info")
                response.raise_for_status()
                return response.json()
        except Exception as error:
            logger.error(f"Error al obtener info del modelo: {error}")
            raise
    
    def clasificar_imagen(
        self, 
        imagen_path: str, 
        filtro: str = "none"
    ) -> Dict[str, any]:
        try:
            logger.info(
                f"Clasificando imagen con filtro: {filtro}"
            )
            
            with open(imagen_path, 'rb') as img_file:
                files = {
                    'file': (
                        Path(imagen_path).name, 
                        img_file, 
                        'image/png'
                    )
                }
                data = {'filter_name': filtro}
                
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(
                        f"{self.base_url}/classify",
                        files=files,
                        data=data
                    )
                    response.raise_for_status()
                    resultado = response.json()
                    
                    logger.info(
                        f"Clasificación recibida: clase "
                        f"{resultado.get('predicted_class')}"
                    )
                    return resultado
        
        except httpx.TimeoutException:
            error_msg = "Tiempo de espera agotado"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        except httpx.HTTPStatusError as error:
            error_msg = f"Error HTTP {error.response.status_code}"
            logger.error(f"{error_msg}: {error}")
            raise RuntimeError(error_msg)
        
        except Exception as error:
            error_msg = f"Error al clasificar imagen: {str(error)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)