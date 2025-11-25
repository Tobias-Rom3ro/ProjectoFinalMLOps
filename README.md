# Pipeline Inteligente MLOps

![Pipeline CI](https://github.com/Tobias-Rom3ro/ProjectoFinalMLOps/actions/workflows/ci.yml/badge.svg)
![Construcción Docker](https://github.com/Tobias-Rom3ro/ProjectoFinalMLOps/actions/workflows/build.yml/badge.svg)
![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)
![Docker](https://img.shields.io/badge/docker-ready-blue.svg)

Sistema integrado de Machine Learning que combina tres servicios independientes:

- **LLM**: Asistente conversacional con Gemini
- **ML Clasico**: Clasificador de vinos con Random Forest
- **CNN**: Clasificador de digitos con Red Neuronal Convolucional

## Estado de los Servicios

| Servicio | Estado | Tests |
|----------|--------|-------|
| LLM Connector | ![Status](https://img.shields.io/badge/status-activo-green) | Pasando |
| Sklearn Model | ![Status](https://img.shields.io/badge/status-activo-green) | Pasando |
| CNN Image | ![Status](https://img.shields.io/badge/status-activo-green) | Pasando |
| Gradio Frontend | ![Status](https://img.shields.io/badge/status-activo-green) | Pasando |

## Inicio Rapido

```bash
# Clonar repositorio
git clone https://github.com/TU_USUARIO/TU_REPOSITORIO.git
cd TU_REPOSITORIO

# Configurar variables de entorno
cp .env.example .env
# Editar .env con tu GENAI_API_KEY

# Iniciar servicios
cd infra
docker-compose up --build
```

## Servicios Disponibles

- MLflow UI: http://localhost:5000
- LLM API: http://localhost:8000/docs
- Sklearn API: http://localhost:8001/docs
- CNN API: http://localhost:8002/docs
- Gradio Frontend: http://localhost:7860

## Arquitectura

```
Gradio Frontend (7860)
    |
    +--- LLM Connector (8000)
    +--- Sklearn Model (8001)
    +--- CNN Image (8002)
    +--- MLflow (5000)
```

## Tecnologias

- Backend: FastAPI, Python 3.10
- ML: scikit-learn, TensorFlow
- LLM: Google Gemini
- Frontend: Gradio
- MLOps: MLflow, Docker
- CI/CD: GitHub Actions

[Resto del README continua aqui...]
