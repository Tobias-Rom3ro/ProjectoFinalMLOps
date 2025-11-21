#!/bin/bash
set -e

echo "========================================"
echo "Inicializando modelo CNN"
echo "========================================"

# Esperar a que MLflow esté disponible
echo "Esperando MLflow..."
until curl -f http://mlflow:5000/api/2.0/mlflow/experiments/search 2>/dev/null; do
    echo "MLflow no disponible, reintentando..."
    sleep 5
done
echo "✓ MLflow disponible"

# Entrenar modelo CNN
echo ""
echo "Entrenando modelo CNN..."
cd /service
python -m pipeline.train
echo "✓ Modelo CNN entrenado"

echo ""
echo "Modelo listo - iniciando servicio..."