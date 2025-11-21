import logging
import gradio as gr
from typing import Dict, Tuple
from app.services.cnn_client import CNNClient
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class CNNInterface:
    def __init__(self, cnn_client: CNNClient):
        self.cnn_client = cnn_client
    
    def clasificar_imagen(
        self, 
        imagen, 
        filtro: str
    ) -> Tuple[np.ndarray, str, Dict[str, float]]:
        """Clasifica imagen y devuelve imagen procesada, resultado y probabilidades."""
        if imagen is None:
            return None, "Por favor, carga una imagen.", {}
        
        try:
            import tempfile
            from pathlib import Path
            
            # Guardar imagen temporal
            with tempfile.NamedTemporaryFile(
                delete=False, 
                suffix='.png'
            ) as tmp_file:
                temp_path = tmp_file.name
                
                # Convertir a PIL Image si es necesario
                if isinstance(imagen, str):
                    img = Image.open(imagen)
                else:
                    img = Image.fromarray(imagen)
                
                # Convertir a escala de grises
                img = img.convert('L')
                img.save(temp_path)
            
            # Clasificar
            resultado = self.cnn_client.clasificar_imagen(
                imagen_path=temp_path,
                filtro=filtro
            )
            
            # Cargar imagen procesada para mostrarla
            img_procesada = Image.open(temp_path)
            img_array = np.array(img_procesada)
            
            # Limpiar archivo temporal
            Path(temp_path).unlink()
            
            # Extraer resultados
            clase_predicha = resultado.get('predicted_class')
            confianza = resultado.get('confidence', 0.0)
            probabilidades = resultado.get('probabilities', {})
            filtro_aplicado = resultado.get('filter_applied', 'none')
            
            # Mensaje de resultado
            mensaje = f"""
## Resultado de Clasificación

**Dígito Predicho:** {clase_predicha}

**Confianza:** {confianza:.2%}

**Filtro Aplicado:** {filtro_aplicado}

### Distribución de Probabilidades
"""
            
            # Preparar datos para el gráfico
            probs_dict = {f"Clase {clase}": prob for clase, prob in probabilidades.items()}
            
            return img_array, mensaje, probs_dict
        
        except Exception as error:
            logger.error(f"Error al clasificar imagen: {error}", exc_info=True)
            return None, f"Error: {str(error)}", {}
    
    def obtener_informacion_modelo(self) -> str:
        try:
            info = self.cnn_client.obtener_info_modelo()
            
            mensaje = f"""
## Información del Modelo CNN

**Tipo:** {info.get('model_type', 'N/A')}

**Tamaño de Entrada:** {info.get('input_size', 'N/A')}

**Número de Clases:** {info.get('num_classes', 'N/A')}

**Descripción:** {info.get('description', 'N/A')}

### Limitaciones:
"""
            
            for limitacion in info.get('limitations', []):
                mensaje += f"\n- {limitacion}"
            
            mensaje += "\n\n### Filtros Disponibles:\n"
            for filtro in info.get('available_filters', []):
                mensaje += f"\n- {filtro}"
            
            return mensaje
        
        except Exception as error:
            logger.error(f"Error al obtener info del modelo: {error}")
            return "Error: No se pudo obtener información del modelo."
    
    def crear_interfaz(self) -> gr.Blocks:
        with gr.Blocks() as interfaz:
            gr.Markdown(
                """
                # Clasificador de Imágenes CNN
                
                Clasifica dígitos escritos a mano (0-9) usando una Red 
                Neuronal Convolucional entrenada con el dataset MNIST.
                
                Puedes aplicar filtros de convolución a la imagen antes 
                de la clasificación.
                """
            )
            
            with gr.Row():
                with gr.Column():
                    imagen_input = gr.Image(
                        label="Sube una imagen de un dígito",
                        type="numpy",
                        height=300
                    )
                    
                    filtro_dropdown = gr.Dropdown(
                        choices=["none", "blur", "edge_detection", "sharpen"],
                        value="none",
                        label="Filtro a aplicar",
                        info="Selecciona un filtro de convolución"
                    )
                    
                    clasificar_btn = gr.Button(
                        "Clasificar Imagen",
                        variant="primary",
                        size="lg"
                    )
                
                with gr.Column():
                    imagen_procesada = gr.Image(
                        label="Imagen Procesada",
                        height=300
                    )
                    
                    resultado_texto = gr.Markdown(label="Resultado")
            
            with gr.Row():
                grafico_probs = gr.BarPlot(
                    x="clase",
                    y="probabilidad",
                    title="Distribución de Probabilidades",
                    height=300
                )
            
            clasificar_btn.click(
                fn=self.clasificar_imagen,
                inputs=[imagen_input, filtro_dropdown],
                outputs=[imagen_procesada, resultado_texto, grafico_probs]
            )
            
            gr.Markdown("---")
            
            info_btn = gr.Button("Ver Información del Modelo")
            info_modelo = gr.Markdown()
            
            info_btn.click(
                fn=self.obtener_informacion_modelo,
                outputs=[info_modelo]
            )
            
            gr.Markdown(
                """
                ### Consejos de Uso
                
                - Sube una imagen clara de un dígito escrito a mano
                - La imagen se redimensionará automáticamente a 28x28 píxeles
                - Los filtros pueden ayudar a mejorar la clasificación
                - El modelo funciona mejor con dígitos centrados y en escala de grises
                """
            )
        
        return interfaz