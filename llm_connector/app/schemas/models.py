from typing import Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    prompt: str = Field(
        ...,
        description="The question or prompt for the LLM",
        min_length=1,
        max_length=10000,
    )
    context: Optional[str] = Field(
        None, description="Optional context for the query", max_length=5000
    )
    model: str = Field(
        "gemini-2.0-flash-exp", description="Model to use for generation"
    )


class QueryResponse(BaseModel):
    response: str = Field(..., description="Generated response from the LLM")
    prompt: str = Field(..., description="Original prompt")
    model_used: str = Field(..., description="Model used for generation")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service health status")
    service: str = Field(..., description="Service name")


class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code")
