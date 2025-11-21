# file: infra/train-models.sh
#!/bin/bash

set -e

echo "========================================"
echo "Entrenando modelos antes del despliegue"
echo "========================================"

echo ""
echo "1. Entrenando modelo sklearn (Wine Classifier)..."
cd ../sklearn_model
python -m pipeline.train_model
echo "Modelo sklearn entrenado correctamente"

echo ""
echo "2. Entrenando modelo CNN (MNIST)..."
cd ../cnn_image
python -m pipeline.train
echo "Modelo CNN entrenado correctamente"

echo ""
echo "========================================"
echo "Todos los modelos entrenados exitosamente"
echo "========================================"