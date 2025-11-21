import pickle
import logging
from pathlib import Path
from typing import List, Optional
import numpy as np
from sklearn.pipeline import Pipeline

from app.config import configuracion

logger = logging.getLogger(__name__)


class GestorModelo:
    """Gestor para cargar y usar el modelo entrenado."""
    
    NOMBRES_CLASES = {
        0: "Clase 0 - Vino tipo A",
        1: "Clase 1 - Vino tipo B",
        2: "Clase 2 - Vino tipo C"
    }
    
    def __init__(self):
        self.modelo: Optional[Pipeline] = None
        self.ruta_modelo: Path = Path(configuracion.ruta_modelo)
        
    def cargar_modelo(self) -> None:
        """Carga el modelo desde el archivo pickle."""
        try:
            if not self.ruta_modelo.exists():
                logger.warning(
                    f"Modelo no encontrado en {self.ruta_modelo}. "
                    "Ejecute el entrenamiento primero."
                )
                return
                
            with open(self.ruta_modelo, 'rb') as archivo:
                self.modelo = pickle.load(archivo)
                
            logger.info(f"Modelo cargado exitosamente desde {self.ruta_modelo}")
            
        except Exception as error:
            logger.error(f"Error al cargar el modelo: {error}")
            raise
            
    def esta_cargado(self) -> bool:
        """Verifica si el modelo está cargado."""
        return self.modelo is not None
        
    def predecir(
        self,
        caracteristicas: List[float]
    ) -> tuple[int, List[float]]:
        """
        Realiza una predicción con el modelo.
        
        Args:
            caracteristicas: Lista de características del vino
            
        Returns:
            Tupla con (clase_predicha, probabilidades)
        """
        if not self.esta_cargado():
            raise RuntimeError(
                "El modelo no está cargado. "
                "Verifique que el entrenamiento se haya ejecutado."
            )
            
        try:
            caracteristicas_array = np.array(caracteristicas).reshape(1, -1)
            
            clase_predicha = self.modelo.predict(caracteristicas_array)[0]
            probabilidades = self.modelo.predict_proba(
                caracteristicas_array
            )[0].tolist()
            
            logger.info(
                f"Predicción realizada: clase={clase_predicha}, "
                f"confianza={max(probabilidades):.3f}"
            )
            
            return int(clase_predicha), probabilidades
            
        except Exception as error:
            logger.error(f"Error en la predicción: {error}")
            raise
            
    def obtener_nombre_clase(self, clase: int) -> str:
        """Obtiene el nombre descriptivo de una clase."""
        return self.NOMBRES_CLASES.get(
            clase,
            f"Clase desconocida ({clase})"
        )


gestor_modelo = GestorModelo()