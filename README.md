# 🤖 Pipeline MLOps / Proyecto Final de Curso.

![Pipeline CI](https://github.com/Tobias-Rom3ro/ProjectoFinalMLOps/actions/workflows/ci.yml/badge.svg)
![Construcción Docker](https://github.com/Tobias-Rom3ro/ProjectoFinalMLOps/actions/workflows/build.yml/badge.svg)
![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)
![Docker](https://img.shields.io/badge/docker-ready-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

> Sistema integrado de Machine Learning que combina tres modelos independientes (LLM, ML Clásico y CNN) en una arquitectura de microservicios con MLOps completo.

## 🎯 Descripción

Este proyecto implementa un **pipeline completo de MLOps** que integra tres servicios de Machine Learning independientes:

1. **🧠 LLM Connector**: Asistente conversacional impulsado por Google Gemini
2. **🍷 Sklearn Model**: Clasificador de vinos usando Random Forest (ML clásico)
3. **🖼️ CNN Image**: Clasificador de dígitos escritos a mano con Red Neuronal Convolucional (MNIST)

Todos los servicios están **containerizados**, incluyen **logging estructurado**, **tracking de experimentos con MLflow**, y se orquestan mediante **Docker Compose** con una interfaz web unificada en **Gradio**.

## ✨ Características

- ✅ **Arquitectura de Microservicios**: Cada modelo ML es un servicio independiente con su propia API REST
- ✅ **MLOps Completo**: Tracking de experimentos, versionado de modelos y registro centralizado con MLflow
- ✅ **Logging Estructurado**: Logs en formato JSON para fácil análisis y debugging
- ✅ **CI/CD Automatizado**: Pipeline completo con GitHub Actions (tests, builds, deployment)
- ✅ **Interfaz Unificada**: Frontend web interactivo con Gradio para todos los servicios
- ✅ **Containerización**: Todos los servicios en Docker con orquestación via Docker Compose
- ✅ **Healthchecks**: Monitoreo de salud de cada servicio
- ✅ **Filtros de Convolución**: Preprocesamiento de imágenes con filtros personalizados (blur, edge detection, sharpen)
- ✅ **Escalabilidad**: Preparado para despliegue en Docker Swarm

## 🏗️ Arquitectura

```
┌─────────────────────────────────────────────────────────────┐
│                    Gradio Frontend (7860)                   │
│              Interfaz Web Unificada de Usuario              │
└─────────────────────┬───────────────────────────────────────┘
                      │
          ┌───────────┼───────────┬────────────┐
          │           │           │            │
          ▼           ▼           ▼            ▼
    ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌──────────┐
    │   LLM   │ │ Sklearn │ │   CNN   │ │  MLflow  │
    │ (8000)  │ │ (8001)  │ │ (8002)  │ │  (5000)  │
    └─────────┘ └─────────┘ └─────────┘ └──────────┘
         │           │           │            │
         └───────────┴───────────┴────────────┘
                      │
              mlops-network (Docker Bridge)
```

### Flujo de Datos

1. **Usuario** → Interactúa con **Gradio Frontend**
2. **Frontend** → Envía requests HTTP a los servicios específicos
3. **Servicios ML** → Procesan requests y devuelven predicciones
4. **MLflow** → Registra métricas, parámetros y artefactos de cada experimento
5. **Logs** → Todos los servicios emiten logs estructurados en JSON

### Componentes Principales

#### 🔌 LLM Connector
- **Framework**: FastAPI
- **Modelo**: Google Gemini 2.5 Flash
- **Función**: Asistente conversacional con contexto
- **Features**: Gestión de historial, contexto multi-turno

#### 🍇 Sklearn Model
- **Framework**: FastAPI + scikit-learn
- **Modelo**: Random Forest Classifier
- **Dataset**: Wine Dataset (178 muestras, 13 características, 3 clases)
- **Pipeline**: StandardScaler → RandomForest
- **Features**: Entrenamiento automático al inicio, predicción con probabilidades

#### 🎨 CNN Image
- **Framework**: FastAPI + TensorFlow/Keras
- **Modelo**: CNN personalizada para MNIST
- **Dataset**: MNIST (dígitos escritos a mano 0-9)
- **Arquitectura**: 2 Conv2D → 2 MaxPooling → Flatten → Dropout → Dense
- **Features**: Filtros de convolución personalizados (blur, edge detection, sharpen)

#### 📊 MLflow
- **Función**: Tracking de experimentos y registro de modelos
- **Storage**: SQLite (backend) + File System (artifacts)
- **Features**: UI web, API REST, versionado de modelos

#### 🖥️ Gradio Frontend
- **Framework**: Gradio
- **Features**: 
  - Chat interactivo con LLM
  - Formulario de clasificación de vinos
  - Upload y clasificación de imágenes con filtros
  - Panel de estado de servicios
  - Visualización de probabilidades

## 🛠️ Tecnologías

### Backend
- **Python 3.10**: Lenguaje principal
- **FastAPI**: Framework web para APIs REST
- **scikit-learn**: ML clásico (Random Forest)
- **TensorFlow/Keras**: Deep Learning (CNN)
- **Google GenAI SDK**: Integración con Gemini

### MLOps
- **MLflow**: Experiment tracking y model registry
- **Docker & Docker Compose**: Containerización y orquestación
- **GitHub Actions**: CI/CD automatizado

### Frontend
- **Gradio**: Interfaz web interactiva

### Infraestructura
- **Docker Networks**: Comunicación entre servicios
- **Docker Volumes**: Persistencia de datos y modelos
- **Healthchecks**: Monitoreo automático de servicios

## 🚀 Inicio Rápido

### Prerequisitos

- Docker 20.10+ y Docker Compose 2.0+
- Git
- API Key de Google Gemini ([Obtener aquí](https://ai.google.dev/))

### Instalación en 3 Pasos

#### 1️⃣ Clonar el Repositorio

```bash
git clone https://github.com/Tobias-Rom3ro/ProjectoFinalMLOps.git
cd ProjectoFinalMLOps
```

#### 2️⃣ Configurar Variables de Entorno

```bash
# Crear archivo .env en la raíz del proyecto
cat > .env << EOF
GENAI_API_KEY=tu_api_key_de_gemini_aqui
LOG_LEVEL=INFO
DEBUG=False
EOF
```

#### 3️⃣ Iniciar Todo el Sistema

```bash
cd infra
docker-compose up --build
```

**¡Eso es todo!** 🎉 El sistema:
- ✅ Construirá todas las imágenes Docker
- ✅ Iniciará MLflow
- ✅ Entrenará los modelos automáticamente
- ✅ Levantará todos los servicios
- ✅ Estará listo en ~3-5 minutos

### Acceso a los Servicios

| Servicio | URL | Descripción |
|----------|-----|-------------|
| 🖥️ **Gradio Frontend** | http://localhost:7860 | Interfaz web principal |
| 📊 **MLflow UI** | http://localhost:5000 | Dashboard de experimentos |
| 🧠 **LLM API Docs** | http://localhost:8000/docs | API Swagger del LLM |
| 🍷 **Sklearn API Docs** | http://localhost:8001/docs | API Swagger del clasificador de vinos |
| 🎨 **CNN API Docs** | http://localhost:8002/docs | API Swagger del clasificador de imágenes |

## 📦 Servicios

### Estado de los Servicios

| Servicio | Puerto | Status | Tests | Health Endpoint |
|----------|--------|--------|-------|-----------------|
| MLflow | 5000 | ![Status](https://img.shields.io/badge/status-activo-green) | N/A | `/health` |
| LLM Connector | 8000 | ![Status](https://img.shields.io/badge/status-activo-green) | ✅ Pasando | `/health` |
| Sklearn Model | 8001 | ![Status](https://img.shields.io/badge/status-activo-green) | ✅ Pasando | `/health` |
| CNN Image | 8002 | ![Status](https://img.shields.io/badge/status-activo-green) | ✅ Pasando | `/health` |
| Gradio Frontend | 7860 | ![Status](https://img.shields.io/badge/status-activo-green) | ✅ Pasando | N/A |

### Verificación de Salud

```bash
# Verificar todos los servicios
curl http://localhost:8000/health  # LLM
curl http://localhost:8001/health  # Sklearn
curl http://localhost:8002/health  # CNN
curl http://localhost:5000/health  # MLflow
```

## 💡 Uso

### 1. Asistente LLM (Chat)

**Interfaz Web:**
1. Abre http://localhost:7860
2. Ve a la pestaña "Asistente LLM"
3. Escribe tu pregunta y presiona Enter

**API REST:**
```bash
curl -X POST http://localhost:8000/llm/query \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "¿Qué es Machine Learning?",
    "context": null,
    "model": "gemini-2.5-flash"
  }'
```

### 2. Clasificador de Vinos

**Interfaz Web:**
1. Ve a la pestaña "Clasificador de Vinos"
2. Ajusta las 13 características químicas
3. Haz clic en "Clasificar Vino"

**API REST:**
```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "alcohol": 13.2,
    "acido_malico": 2.3,
    "ceniza": 2.4,
    "alcalinidad_ceniza": 19.5,
    "magnesio": 100.0,
    "fenoles_totales": 2.8,
    "flavonoides": 3.0,
    "fenoles_no_flavonoides": 0.3,
    "proantocianinas": 1.9,
    "intensidad_color": 5.6,
    "matiz": 1.0,
    "od280_od315": 3.2,
    "prolina": 1050.0
  }'
```

### 3. Clasificador de Imágenes (CNN)

**Interfaz Web:**
1. Ve a la pestaña "Clasificador de Imágenes"
2. Sube una imagen de un dígito (0-9)
3. Opcionalmente selecciona un filtro (blur, edge detection, sharpen)
4. Haz clic en "Clasificar Imagen"

**API REST:**
```bash
curl -X POST http://localhost:8002/classify \
  -F "file=@digit.png" \
  -F "filter_name=edge_detection"
```

### 4. MLflow UI

1. Abre http://localhost:5000
2. Explora experimentos:
   - `sklearn_wine_classifier`
   - `cnn_mnist_classification`
3. Compara runs, métricas y parámetros
4. Descarga modelos registrados

## 🔄 CI/CD

### Pipeline Automatizado

El proyecto incluye dos workflows de GitHub Actions:

#### 1. **Pipeline CI** (`.github/workflows/ci.yml`)
- **Trigger**: Push/PR a `main` o `develop`
- **Jobs**:
  - ✅ Tests de LLM Connector
  - ✅ Tests de Sklearn Model
  - ✅ Tests de CNN Image
  - ✅ Tests de Gradio Frontend
- **Python**: 3.10
- **Caché**: Dependencias pip

#### 2. **Build Docker** (`.github/workflows/build.yml`)
- **Trigger**: Push a `main` o manual
- **Jobs**:
  - 🏗️ Build de todas las imágenes Docker
  - 📦 Push a GitHub Container Registry (ghcr.io)
  - 🏷️ Tags: `latest`, `sha-{commit}`, `branch-{branch}`
- **Registry**: `ghcr.io/tobias-rom3ro/projectofinalmlops/`

### Ejecutar Tests Localmente

```bash
# Todos los tests
pytest

# Tests de un servicio específico
pytest llm_connector/tests/ -v
pytest sklearn_model/tests/ -v
pytest cnn_image/tests/ -v
pytest gradio_frontend/tests/ -v
```

## 🚢 Despliegue

### Docker Compose (Local/Dev)

```bash
# Iniciar todos los servicios
cd infra
docker-compose up -d

# Ver logs
docker-compose logs -f

# Ver logs de un servicio específico
docker-compose logs -f sklearn_model

# Detener servicios
docker-compose down

# Detener y eliminar volúmenes
docker-compose down -v
```

### Docker Swarm (Producción)

```bash
# Inicializar Swarm (si no está activo)
docker swarm init

# Desplegar stack
cd infra
docker stack deploy -c swarm-stack.yml mlops-final-project

# Ver servicios
docker stack services mlops-final-project

# Ver logs de un servicio
docker service logs mlops-final-project_sklearn_model

# Escalar un servicio
docker service scale mlops-final-project_llm_connector=5

# Eliminar stack
docker stack rm mlops-final-project
```

**Características del despliegue en Swarm:**
- 🔄 **Replicas**: LLM (2), Sklearn (2), CNN (2), Gradio (1), MLflow (1)
- 🔄 **Restart Policy**: Automático con 3 reintentos
- 💾 **Resource Limits**: CPU y memoria limitados por servicio
- 🌐 **Network**: Overlay network para comunicación multi-host
- 📦 **Volumes**: Persistencia de datos y modelos

## 📊 Estructura del Proyecto

```
.
├── .github/
│   └── workflows/          # GitHub Actions CI/CD
│       ├── ci.yml
│       └── build.yml
├── infra/
│   ├── mlflow/             # Configuración MLflow
│   ├── scripts/            # Scripts de utilidad
│   ├── docker-compose.yml  # Orquestación local
│   └── swarm-stack.yml     # Orquestación producción
├── llm_connector/          # Servicio LLM
│   ├── app/
│   │   ├── api/
│   │   ├── core/
│   │   ├── schemas/
│   │   └── services/
│   ├── tests/
│   ├── Dockerfile
│   └── requirements.txt
├── sklearn_model/          # Servicio ML Clásico
│   ├── app/
│   ├── pipeline/           # Training pipeline
│   ├── tests/
│   ├── Dockerfile
│   ├── init-models.sh      # Auto-training script
│   └── requirements.txt
├── cnn_image/              # Servicio CNN
│   ├── app/
│   ├── filters/            # Filtros de convolución
│   ├── pipeline/           # Training pipeline
│   ├── tests/
│   ├── Dockerfile
│   ├── init-models.sh
│   └── requirements.txt
├── gradio_frontend/        # Frontend Web
│   ├── app/
│   │   ├── services/       # Clientes HTTP
│   │   └── ui/             # Interfaces Gradio
│   ├── tests/
│   ├── Dockerfile
│   └── requirements.txt
├── .env.example            # Ejemplo de configuración
├── .gitignore
├── pyproject.toml          # Configuración pytest
└── README.md
```

## 🧪 Testing

### Cobertura de Tests

| Servicio | Archivos | Tests | Cobertura |
|----------|----------|-------|-----------|
| LLM Connector | 2 | 7 tests | Core functionality |
| Sklearn Model | 1 | 4 tests | Endpoints + Model |
| CNN Image | 3 | 12 tests | Endpoints + Filters + Model |
| Gradio Frontend | 2 | 10 tests | Imports + Clients |

### Ejecutar Tests con Coverage

```bash
# Instalar coverage
pip install pytest-cov

# Ejecutar con reporte
pytest --cov=app --cov-report=html

# Ver reporte
open htmlcov/index.html
```

## 🐛 Troubleshooting

### Problema: Servicios no inician
```bash
# Verificar logs
docker-compose logs

# Verificar que el puerto no esté ocupado
lsof -i :8000  # o el puerto que sea
```

### Problema: MLflow no conecta
```bash
# Verificar que MLflow esté corriendo
curl http://localhost:5000/health

# Reiniciar MLflow
docker-compose restart mlflow
```

### Problema: Modelos no se entrenan
```bash
# Verificar logs del entrenamiento
docker-compose logs sklearn_model
docker-compose logs cnn_image

# Entrenar manualmente
docker-compose exec sklearn_model python -m pipeline.train
docker-compose exec cnn_image python -m pipeline.train
```

### Problema: API Key de Gemini inválida
```bash
# Verificar archivo .env
cat .env

# Actualizar y reiniciar
docker-compose restart llm_connector
```

## 📝 Variables de Entorno

| Variable | Descripción | Default | Requerida |
|----------|-------------|---------|-----------|
| `GENAI_API_KEY` | API Key de Google Gemini | - | ✅ Sí |
| `LOG_LEVEL` | Nivel de logging | INFO | ❌ No |
| `DEBUG` | Modo debug | False | ❌ No |
| `MLFLOW_TRACKING_URI` | URL de MLflow | http://mlflow:5000 | ❌ No |

## 👥 Autores

- **Tobias Romero** - [GitHub](https://github.com/Tobias-Rom3ro)
- **Jenifer Roa**

---

⭐ **Si este proyecto te fue útil, considera darle una estrella!** ⭐
