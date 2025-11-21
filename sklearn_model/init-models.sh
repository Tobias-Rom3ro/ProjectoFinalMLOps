#!/bin/bash
set -e

echo "========================================"
echo "Inicializando y entrenando modelos"
echo "========================================"

# Esperar a que MLflow esté disponible
echo "Esperando MLflow..."
until curl -f http://mlflow:5000/api/2.0/mlflow/experiments/search 2>/dev/null; do
    echo "MLflow no disponible, reintentando..."
    sleep 5
done
echo "✓ MLflow disponible"

# Entrenar modelo sklearn
echo ""
echo "Entrenando modelo sklearn..."
cd /service
python -m pipeline.train
echo "✓ Modelo sklearn entrenado"

echo ""
echo "Modelos listos - iniciando servicio..."