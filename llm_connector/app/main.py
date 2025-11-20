import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.logging_config import setup_logging
from app.api.routes import router
from app.api.dependencies import initialize_gemini_service


settings = get_settings()
setup_logging(settings.service_name, settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.service_name} service")

    try:
        initialize_gemini_service()
        logger.info("Gemini service initialized successfully")
    except Exception as error:
        logger.error(f"Failed to initialize service: {error}")
        raise

    yield

    logger.info(f"Shutting down {settings.service_name} service")


app = FastAPI(
    title="LLM Connector Service",
    description="Service to interact with Google Gemini LLM",
    version="1.0.0",
    lifespan=lifespan,
)


app.include_router(router)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
