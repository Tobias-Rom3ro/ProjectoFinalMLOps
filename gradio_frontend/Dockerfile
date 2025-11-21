# file: gradio_frontend/Dockerfile
FROM python:3.10-slim

WORKDIR /service

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE 7860

CMD ["python", "-m", "app.main"]