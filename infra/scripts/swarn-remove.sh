# file: infra/swarm-remove.sh
#!/bin/bash

set -e

echo "========================================"
echo "Eliminando stack de Swarm"
echo "========================================"

docker stack rm mlops-final-project

echo ""
echo "Esperando a que se eliminen los servicios..."
sleep 10

echo ""
echo "Stack eliminado correctamente"