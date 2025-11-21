# file: infra/start-services.sh
#!/bin/bash

set -e

echo "========================================"
echo "Iniciando Pipeline Inteligente MLOps"
echo "========================================"

echo ""
echo "Verificando archivo .env..."
if [ ! -f ../.env ]; then
    echo "ERROR: Archivo .env no encontrado en la raíz del proyecto"
    echo "Por favor, crea el archivo .env con las variables necesarias"
    exit 1
fi

echo ""
echo "Construyendo e iniciando servicios con Docker Compose..."
docker-compose up --build -d

echo ""
echo "Esperando a que los servicios estén listos..."
sleep 10

echo ""
echo "========================================"
echo "Estado de los servicios:"
echo "========================================"
docker-compose ps

echo ""
echo "========================================"
echo "Servicios disponibles en:"
echo "========================================"
echo "- MLflow UI:        http://localhost:5000"
echo "- LLM Connector:    http://localhost:8000"
echo "- Sklearn Model:    http://localhost:8001"
echo "- CNN Image:        http://localhost:8002"
echo "- Gradio Frontend:  http://localhost:7860"
echo "========================================"