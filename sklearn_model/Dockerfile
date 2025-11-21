# file: sklearn_model/Dockerfile
FROM python:3.10-slim

WORKDIR /service

# Instalar dependencias del sistema necesarias para compilar paquetes científicos
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY pipeline/ ./pipeline/

# Crear directorios necesarios
RUN mkdir -p models data

EXPOSE 8001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]