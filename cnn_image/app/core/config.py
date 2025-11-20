from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    service_name: str = "cnn_image"
    log_level: str = "INFO"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8002
    
    mlflow_tracking_uri: str = "http://mlflow:5000"
    mlflow_experiment_name: str = "cnn_mnist_classification"
    
    cnn_model_path: str = "models/mnist_cnn_model.keras"
    image_size: int = 28
    num_classes: int = 10
    
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        protected_namespaces=('settings_',)
    )


def get_settings() -> Settings:
    return Settings()