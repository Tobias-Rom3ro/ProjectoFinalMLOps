import logging
from fastapi import APIRouter, HTTPException, status, Depends

from app.schemas.models import (
    QueryRequest, 
    QueryResponse, 
    HealthResponse,
    ErrorResponse
)
from app.services.gemini_service import GeminiService
from app.api.dependencies import get_gemini_service
from app.core.config import get_settings


logger = logging.getLogger(__name__)
router = APIRouter()


@router.get(
    "/health", 
    response_model=HealthResponse, 
    status_code=status.HTTP_200_OK,
    tags=["Health"]
)
async def health_check():
    settings = get_settings()
    logger.info("Health check endpoint called")
    return HealthResponse(
        status="healthy", 
        service=settings.service_name
    )


@router.post(
    "/llm/query", 
    response_model=QueryResponse, 
    status_code=status.HTTP_200_OK,
    responses={
        503: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    tags=["LLM"]
)
async def query_llm(
    request: QueryRequest,
    gemini_service: GeminiService = Depends(get_gemini_service)
):
    logger.info(
        f"Received query request - prompt length: {len(request.prompt)}, "
        f"has_context: {request.context is not None}"
    )
    
    if not gemini_service.is_available():
        logger.error("Gemini service not available")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM service is not available"
        )
    
    try:
        response_text = gemini_service.generate_response(
            prompt=request.prompt,
            context=request.context,
            model=request.model
        )
        
        logger.info("Query processed successfully")
        return QueryResponse(
            response=response_text,
            prompt=request.prompt,
            model_used=request.model
        )
    
    except RuntimeError as error:
        logger.error(f"Runtime error processing query: {error}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(error)
        )
    
    except Exception as error:
        logger.error(f"Unexpected error processing query: {error}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred"
        )