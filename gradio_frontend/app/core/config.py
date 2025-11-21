from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    service_name: str = "gradio_frontend"
    log_level: str = "INFO"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 7860
    
    llm_connector_url: str = "http://llm_connector:8000"
    sklearn_model_url: str = "http://sklearn_model:8001"
    cnn_image_url: str = "http://cnn_image:8002"
    
    request_timeout: int = 30
    
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False
    )


def get_settings() -> Settings:
    return Settings()