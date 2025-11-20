from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    service_name: str = "cnn_image"
    log_level: str = "INFO"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8002
    
    mlflow_tracking_uri: str = ""
    model_path: str = "./models/saved_model.pth"
    
    input_size: int = 28
    num_classes: int = 10
    
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False
    )


def get_settings() -> Settings:
    return Settings()