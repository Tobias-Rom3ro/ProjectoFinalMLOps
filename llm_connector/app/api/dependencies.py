from typing import Generator, Optional
from app.services.gemini_service import GeminiService
from app.core.config import get_settings


_gemini_service: Optional[GeminiService] = None


def initialize_gemini_service() -> None:
    global _gemini_service
    settings = get_settings()
    _gemini_service = GeminiService(api_key=settings.genai_api_key)


def get_gemini_service() -> Generator[GeminiService, None, None]:
    if _gemini_service is None:
        raise RuntimeError("Gemini service not initialized")
    yield _gemini_service
