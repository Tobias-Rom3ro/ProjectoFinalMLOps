import logging
from typing import Optional, Dict
import httpx

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        logger.info(f"LLM Client inicializado con URL: {self.base_url}")
    
    def verificar_salud(self) -> Dict[str, any]:
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(f"{self.base_url}/health")
                response.raise_for_status()
                return response.json()
        except Exception as error:
            logger.error(f"Error al verificar salud del LLM: {error}")
            raise
    
    def consultar(
        self, 
        pregunta: str, 
        contexto: Optional[str] = None
    ) -> str:
        try:
            payload = {
                "prompt": pregunta,
                "context": contexto,
                "model": "gemini-2.0-flash-exp"
            }
            
            logger.info(f"Enviando consulta al LLM: {pregunta[:50]}...")
            
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.base_url}/llm/query",
                    json=payload
                )
                response.raise_for_status()
                data = response.json()
                
                logger.info("Respuesta recibida del LLM")
                return data.get("response", "Sin respuesta")
        
        except httpx.TimeoutException:
            error_msg = "Tiempo de espera agotado al consultar el LLM"
            logger.error(error_msg)
            return f"Error: {error_msg}"
        
        except httpx.HTTPStatusError as error:
            error_msg = f"Error HTTP {error.response.status_code}"
            logger.error(f"{error_msg}: {error}")
            return f"Error: {error_msg}"
        
        except Exception as error:
            error_msg = f"Error al consultar el LLM: {str(error)}"
            logger.error(error_msg)
            return f"Error: {error_msg}"