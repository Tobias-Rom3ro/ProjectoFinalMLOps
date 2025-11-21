# file: infra/logs.sh
#!/bin/bash

if [ -z "$1" ]; then
    echo "Mostrando logs de todos los servicios..."
    docker-compose logs -f
else
    echo "Mostrando logs del servicio: $1"
    docker-compose logs -f $1
fi