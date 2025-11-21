FROM python:3.10-slim

WORKDIR /service

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
COPY init-models.sh /init-models.sh

RUN mkdir -p models data && \
    chmod +x /init-models.sh

EXPOSE 8001

# Script que entrena el modelo antes de iniciar el servicio
CMD ["/bin/bash", "-c", "/init-models.sh && uvicorn app.main:app --host 0.0.0.0 --port 8001"]