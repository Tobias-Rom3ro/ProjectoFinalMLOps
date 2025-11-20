"""
Script para analizar el rendimiento del modelo en detalle.
"""
import pickle
import numpy as np
from pathlib import Path
from sklearn.datasets import load_wine
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

from app.config import configuracion, configurar_logging
from pipeline.utils import dividir_datos

import logging
logger = logging.getLogger(__name__)


def analizar_modelo_detallado():
    """Analiza el modelo en profundidad con el conjunto de prueba."""
    configurar_logging()
    logger.info("Iniciando análisis detallado del modelo")
    
    ruta_modelo = Path(configuracion.ruta_modelo)
    
    if not ruta_modelo.exists():
        logger.error(f"Modelo no encontrado en {ruta_modelo}")
        return
    
    with open(ruta_modelo, 'rb') as archivo:
        modelo = pickle.load(archivo)
    
    datos = load_wine()
    X, y = datos.data, datos.target
    nombres_clases = datos.target_names
    
    X_train, X_test, y_train, y_test = dividir_datos(X, y)
    
    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)
    
    print("\n" + "="*70)
    print("ANÁLISIS DETALLADO DEL MODELO")
    print("="*70)
    
    print("\n1. REPORTE DE CLASIFICACIÓN:")
    print("-" * 70)
    print(classification_report(
        y_test,
        y_pred,
        target_names=nombres_clases,
        digits=3
    ))
    
    print("\n2. MATRIZ DE CONFUSIÓN:")
    print("-" * 70)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("\nInterpretación:")
    for i, nombre_clase in enumerate(nombres_clases):
        correctos = cm[i, i]
        total = cm[i, :].sum()
        precision_clase = correctos / total if total > 0 else 0
        print(f"  {nombre_clase}: {correctos}/{total} correctos "
              f"({precision_clase:.1%})")
    
    print("\n3. CASOS CON BAJA CONFIANZA (< 70%):")
    print("-" * 70)
    casos_baja_confianza = []
    for idx, (prediccion, real, probabilidades) in enumerate(
        zip(y_pred, y_test, y_proba)
    ):
        confianza = max(probabilidades)
        if confianza < 0.70:
            casos_baja_confianza.append({
                'indice': idx,
                'real': real,
                'prediccion': prediccion,
                'confianza': confianza,
                'probabilidades': probabilidades
            })
    
    if casos_baja_confianza:
        for caso in casos_baja_confianza[:5]:
            print(f"\n  Muestra {caso['indice']}:")
            print(f"    Real: {nombres_clases[caso['real']]}")
            print(f"    Predicho: {nombres_clases[caso['prediccion']]}")
            print(f"    Confianza: {caso['confianza']:.1%}")
            print(f"    Probabilidades: {caso['probabilidades']}")
    else:
        print("No hay casos con baja confianza")
    
    print("\n4. ERRORES DE CLASIFICACIÓN:")
    print("-" * 70)
    errores = np.where(y_pred != y_test)[0]
    print(f"Total de errores: {len(errores)}/{len(y_test)} "
          f"({len(errores)/len(y_test):.1%})")
    
    if len(errores) > 0:
        print("\nPrimeros 5 errores:")
        for idx in errores[:5]:
            print(f"\n  Muestra {idx}:")
            print(f"    Clase real: {nombres_clases[y_test[idx]]}")
            print(f"    Predicción: {nombres_clases[y_pred[idx]]}")
            print(f"    Confianza: {max(y_proba[idx]):.1%}")
            print(f"    Características: {X_test[idx][:5]}...")
    
    print("\n5. ANÁLISIS POR CLASE:")
    print("-" * 70)
    for i, nombre_clase in enumerate(nombres_clases):
        indices_clase = np.where(y_test == i)[0]
        if len(indices_clase) > 0:
            predicciones_clase = y_pred[indices_clase]
            confianzas_clase = y_proba[indices_clase].max(axis=1)
            
            correctos = (predicciones_clase == i).sum()
            precision = correctos / len(indices_clase)
            confianza_promedio = confianzas_clase.mean()
            
            print(f"\n  {nombre_clase}:")
            print(f"    Muestras en test: {len(indices_clase)}")
            print(f"    Correctamente clasificadas: {correctos} "
                  f"({precision:.1%})")
            print(f"    Confianza promedio: {confianza_promedio:.1%}")
            
            if precision < 1.0:
                confusiones = []
                for j in range(len(nombres_clases)):
                    if j != i:
                        confundidos = (predicciones_clase == j).sum()
                        if confundidos > 0:
                            confusiones.append(
                                f"{nombres_clases[j]} ({confundidos})"
                            )
                if confusiones:
                    print(f"    Confundida con: {', '.join(confusiones)}")
    
    print("\n" + "="*70)
    print("CONCLUSIÓN:")
    print("-" * 70)
    
    precision_global = (y_pred == y_test).sum() / len(y_test)
    
    if precision_global >= 0.90:
        print(" Excelente rendimiento del modelo")
    elif precision_global >= 0.80:
        print("Buen rendimiento del modelo")
    elif precision_global >= 0.70:
        print("Rendimiento aceptable, considere ajustar hiperparámetros")
    else:
        print("Rendimiento bajo, se recomienda reentrenar")
    
    print(f"\nPrecisión global: {precision_global:.1%}")
    print("="*70 + "\n")


if __name__ == "__main__":
    analizar_modelo_detallado()