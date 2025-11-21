FROM python:3.10-slim

WORKDIR /mlflow

RUN apt-get update && apt-get install -y \
    curl \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    mlflow==3.6.0 \
    psutil==5.9.8

# Crear directorios con permisos correctos
RUN mkdir -p /mlflow/backend /mlflow/artifacts && \
    chmod -R 777 /mlflow/backend /mlflow/artifacts

EXPOSE 5000

CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:////mlflow/backend/mlflow.db", \
     "--default-artifact-root", "/mlflow/artifacts", \
     "--host", "0.0.0.0", \
     "--port", "5000"]