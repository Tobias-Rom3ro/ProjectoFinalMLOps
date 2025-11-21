from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint():
    """Verifica que el endpoint de salud responda correctamente."""
    respuesta = client.get("/health")
    
    assert respuesta.status_code == 200
    datos = respuesta.json()
    
    assert "estado" in datos
    assert "modelo_cargado" in datos
    assert "mlflow_conectado" in datos
    assert "version" in datos


def test_predict_endpoint_estructura():
    """Verifica que el endpoint de predicción tenga la estructura correcta."""
    datos_prueba = {
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
    }
    
    respuesta = client.post("/predict", json=datos_prueba)
    
    assert respuesta.status_code in [200, 503]
    
    if respuesta.status_code == 200:
        datos = respuesta.json()
        assert "clase_predicha" in datos
        assert "nombre_clase" in datos
        assert "probabilidades" in datos
        assert "confianza" in datos
        assert isinstance(datos["probabilidades"], list)
        assert len(datos["probabilidades"]) == 3
        assert 0 <= datos["confianza"] <= 1


def test_predict_endpoint_validacion():
    """Verifica que el endpoint valide datos incorrectos."""
    datos_invalidos = {
        "alcohol": -5.0,
        "acido_malico": 2.3,
        "ceniza": 2.4
    }
    
    respuesta = client.post("/predict", json=datos_invalidos)
    
    assert respuesta.status_code == 422


def test_predict_endpoint_sin_modelo():
    """Verifica el comportamiento cuando el modelo no está cargado."""
    from app.model import gestor_modelo
    
    modelo_original = gestor_modelo.modelo
    gestor_modelo.modelo = None
    
    datos_prueba = {
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
    }
    
    respuesta = client.post("/predict", json=datos_prueba)
    
    assert respuesta.status_code == 503
    
    gestor_modelo.modelo = modelo_original