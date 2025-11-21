# file: infra/swarm-deploy.sh
#!/bin/bash

set -e

echo "========================================"
echo "Desplegando en Docker Swarm"
echo "========================================"

echo ""
echo "1. Verificando Swarm..."
if ! docker info | grep -q "Swarm: active"; then
    echo "Inicializando Docker Swarm..."
    docker swarm init
else
    echo "Swarm ya está activo"
fi

echo ""
echo "2. Construyendo imágenes..."
cd ..
docker-compose -f infra/docker-compose.yml build

echo ""
echo "3. Etiquetando imágenes para Swarm..."
docker tag mlops-final-project_mlflow:latest mlops-final-project_mlflow:latest
docker tag mlops-final-project_llm_connector:latest mlops-final-project_llm_connector:latest
docker tag mlops-final-project_sklearn_model:latest mlops-final-project_sklearn_model:latest
docker tag mlops-final-project_cnn_image:latest mlops-final-project_cnn_image:latest
docker tag mlops-final-project_gradio_frontend:latest mlops-final-project_gradio_frontend:latest

echo ""
echo "4. Desplegando stack en Swarm..."
docker stack deploy -c infra/swarm-stack.yml mlops-final-project

echo ""
echo "5. Esperando despliegue..."
sleep 15

echo ""
echo "========================================"
echo "Estado del stack:"
echo "========================================"
docker stack services mlops-final-project

echo ""
echo "========================================"
echo "Stack desplegado exitosamente"
echo "========================================"