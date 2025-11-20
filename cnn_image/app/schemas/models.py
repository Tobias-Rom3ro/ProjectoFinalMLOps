from typing import Optional, Dict, List
from pydantic import BaseModel, Field

class ClassificationRequest(BaseModel):
    filter_name: str = Field(
        "none",
        description="Filter to apply before classification"
    )


class ClassificationResponse(BaseModel):
    predicted_class: int = Field(..., description="Predicted class label")
    confidence: float = Field(..., description="Confidence score")
    probabilities: Dict[str, float] = Field(
        ..., 
        description="Probability distribution across all classes"
    )
    filter_applied: str = Field(..., description="Filter that was applied")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service health status")
    service: str = Field(..., description="Service name")
    model_loaded: bool = Field(..., description="Whether model is loaded")


class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code")


class ModelInfoResponse(BaseModel):
    model_type: str
    input_size: str
    num_classes: int
    classes: List[str]
    description: str
    limitations: List[str]
    available_filters: List[str]