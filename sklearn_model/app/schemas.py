from pydantic import BaseModel, Field
from typing import List


class CaracteristicasVino(BaseModel):
    """Esquema para las características del vino."""
    
    alcohol: float = Field(
        ...,
        description="Contenido de alcohol",
        ge=0.0,
        le=20.0
    )
    acido_malico: float = Field(
        ...,
        description="Ácido málico",
        ge=0.0
    )
    ceniza: float = Field(
        ...,
        description="Contenido de ceniza",
        ge=0.0
    )
    alcalinidad_ceniza: float = Field(
        ...,
        description="Alcalinidad de la ceniza",
        ge=0.0
    )
    magnesio: float = Field(
        ...,
        description="Contenido de magnesio",
        ge=0.0
    )
    fenoles_totales: float = Field(
        ...,
        description="Fenoles totales",
        ge=0.0
    )
    flavonoides: float = Field(
        ...,
        description="Contenido de flavonoides",
        ge=0.0
    )
    fenoles_no_flavonoides: float = Field(
        ...,
        description="Fenoles no flavonoides",
        ge=0.0
    )
    proantocianinas: float = Field(
        ...,
        description="Proantocianinas",
        ge=0.0
    )
    intensidad_color: float = Field(
        ...,
        description="Intensidad del color",
        ge=0.0
    )
    matiz: float = Field(
        ...,
        description="Matiz",
        ge=0.0
    )
    od280_od315: float = Field(
        ...,
        description="Relación OD280/OD315 de vinos diluidos",
        ge=0.0
    )
    prolina: float = Field(
        ...,
        description="Contenido de prolina",
        ge=0.0
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "alcohol": 13.2,
                "acido_malico": 2.3,
                "ceniza": 2.4,
                "alcalinidad_ceniza": 19.5,
                "magnesio": 100.0,
                "fenoles_totales": 2.8,
                "flavonoides": 3.0,
                "fenoles_no_flavonoides": 0.3,
                "proantocianinas": 1.9,
                "intensidad_color": 5.6,
                "matiz": 1.0,
                "od280_od315": 3.2,
                "prolina": 1050.0
            }
        }


class PrediccionRespuesta(BaseModel):
    """Esquema para la respuesta de predicción."""
    
    clase_predicha: int = Field(
        ...,
        description="Clase de vino predicha (0, 1, o 2)"
    )
    nombre_clase: str = Field(
        ...,
        description="Nombre descriptivo de la clase"
    )
    probabilidades: List[float] = Field(
        ...,
        description="Probabilidades para cada clase"
    )
    confianza: float = Field(
        ...,
        description="Confianza de la predicción (probabilidad máxima)",
        ge=0.0,
        le=1.0
    )


class EstadoSalud(BaseModel):
    """Esquema para el estado de salud del servicio."""
    
    estado: str
    modelo_cargado: bool
    mlflow_conectado: bool
    version: str = "1.0.0"