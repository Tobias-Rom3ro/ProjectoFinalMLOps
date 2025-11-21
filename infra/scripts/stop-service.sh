# file: infra/stop-services.sh
#!/bin/bash

set -e

echo "========================================"
echo "Deteniendo servicios"
echo "========================================"

docker-compose down

echo ""
echo "Servicios detenidos correctamente"