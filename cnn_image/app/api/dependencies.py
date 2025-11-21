from typing import Generator
from app.services.cnn_service import CNNService
from app.services.filter_service import FilterService
from app.core.config import get_settings


_cnn_service: CNNService = None
_filter_service: FilterService = None


def initialize_services() -> None:
    global _cnn_service, _filter_service
    settings = get_settings()
    
    _cnn_service = CNNService(
        cnn_model_path=settings.cnn_model_path,
        image_size=settings.image_size,
        num_classes=settings.num_classes
    )
    
    _filter_service = FilterService()


def get_cnn_service() -> Generator[CNNService, None, None]:
    if _cnn_service is None:
        raise RuntimeError("CNN service not initialized")
    yield _cnn_service


def get_filter_service() -> Generator[FilterService, None, None]:
    if _filter_service is None:
        raise RuntimeError("Filter service not initialized")
    yield _filter_service