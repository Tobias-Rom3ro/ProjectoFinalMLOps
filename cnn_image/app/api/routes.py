import logging
import io
from fastapi import (
    APIRouter, 
    HTTPException, 
    status, 
    Depends, 
    UploadFile, 
    File,
    Form
)
from PIL import Image

from app.schemas.models import (
    ClassificationResponse,
    HealthResponse,
    ErrorResponse,
    ModelInfoResponse
)
from app.services.cnn_service import CNNService
from app.services.filter_service import FilterService
from app.api.dependencies import get_cnn_service, get_filter_service
from app.core.config import get_settings


logger = logging.getLogger(__name__)
router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    tags=["Health"]
)
async def health_check(
    cnn_service: CNNService = Depends(get_cnn_service)
):
    settings = get_settings()
    logger.info("Health check endpoint called")
    return HealthResponse(
        status="healthy",
        service=settings.service_name,
        model_loaded=cnn_service.is_available()
    )


@router.get(
    "/model/info",
    response_model=ModelInfoResponse,
    status_code=status.HTTP_200_OK,
    tags=["Model"]
)
async def get_model_info(
    cnn_service: CNNService = Depends(get_cnn_service),
    filter_service: FilterService = Depends(get_filter_service)
):
    logger.info("Model info endpoint called")
    
    model_info = cnn_service.get_model_info()
    model_info["available_filters"] = filter_service.get_available_filters()
    
    return ModelInfoResponse(**model_info)


@router.post(
    "/classify",
    response_model=ClassificationResponse,
    status_code=status.HTTP_200_OK,
    responses={
        503: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    tags=["Classification"]
)
async def classify_image(
    file: UploadFile = File(...),
    filter_name: str = Form("none"),
    cnn_service: CNNService = Depends(get_cnn_service),
    filter_service: FilterService = Depends(get_filter_service)
):
    logger.info(f"Classification request with filter: {filter_name}")
    
    if not cnn_service.is_available():
        logger.error("CNN model not available")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not available. Please train the model first."
        )
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        logger.info(
            f"Image loaded: size={image.size}, mode={image.mode}"
        )
        
        if filter_name != "none":
            image = filter_service.apply_filter(image, filter_name)
            logger.info(f"Filter '{filter_name}' applied")
        
        prediction_result = cnn_service.predict(image)
        
        logger.info("Classification completed successfully")
        return ClassificationResponse(
            predicted_class=prediction_result["predicted_class"],
            confidence=prediction_result["confidence"],
            probabilities=prediction_result["probabilities"],
            filter_applied=filter_name
        )
    
    except Exception as error:
        logger.error(f"Error during classification: {error}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Classification failed: {str(error)}"
        )