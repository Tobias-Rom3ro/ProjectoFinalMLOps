#!/bin/bash
set -e

echo "========================================"
echo "Inicializando y entrenando modelos"
echo "========================================"

echo "Esperando MLflow..."
until curl -fs -X POST http://mlflow:5000/api/2.0/mlflow/experiments/search \
  -H "Content-Type: application/json" \
  -d '{"max_results": 1}' >/dev/null; do
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