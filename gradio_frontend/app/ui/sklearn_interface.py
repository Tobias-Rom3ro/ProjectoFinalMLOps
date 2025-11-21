import logging
import gradio as gr
from app.services.sklearn_client import SklearnClient

logger = logging.getLogger(__name__)


class SklearnInterface:
    def __init__(self, sklearn_client: SklearnClient):
        self.sklearn_client = sklearn_client
    
    def realizar_prediccion(
        self,
        alcohol: float,
        acido_malico: float,
        ceniza: float,
        alcalinidad_ceniza: float,
        magnesio: float,
        fenoles_totales: float,
        flavonoides: float,
        fenoles_no_flavonoides: float,
        proantocianinas: float,
        intensidad_color: float,
        matiz: float,
        od280_od315: float,
        prolina: float
    ) -> str:
        try:
            resultado = self.sklearn_client.predecir(
                alcohol=alcohol,
                acido_malico=acido_malico,
                ceniza=ceniza,
                alcalinidad_ceniza=alcalinidad_ceniza,
                magnesio=magnesio,
                fenoles_totales=fenoles_totales,
                flavonoides=flavonoides,
                fenoles_no_flavonoides=fenoles_no_flavonoides,
                proantocianinas=proantocianinas,
                intensidad_color=intensidad_color,
                matiz=matiz,
                od280_od315=od280_od315,
                prolina=prolina
            )
            
            clase = resultado.get('clase_predicha')
            nombre_clase = resultado.get('nombre_clase', 'Desconocido')
            confianza = resultado.get('confianza', 0.0)
            probabilidades = resultado.get('probabilidades', [])
            
            mensaje = f"""
            ## Resultado de Clasificación
            
            **Clase Predicha:** {clase} ({nombre_clase})
            
            **Confianza:** {confianza:.2%}
            
            ### Probabilidades por Clase:
            """
            
            for i, prob in enumerate(probabilidades):
                mensaje += f"\n- Clase {i}: {prob:.2%}"
            
            return mensaje
        
        except Exception as error:
            logger.error(f"Error en predicción: {error}")
            return f"Error al realizar la predicción: {str(error)}"
    
    def crear_interfaz(self) -> gr.Blocks:
        with gr.Blocks() as interfaz:
            gr.Markdown(
                """
                # Clasificador de Vinos
                
                Ingresa las características químicas de un vino para 
                clasificarlo en una de las tres categorías disponibles.
                
                **Modelo:** Random Forest entrenado con dataset Wine de scikit-learn
                
                **Clases:** 0, 1, 2 (variedades de vino)
                """
            )
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Características Principales")
                    
                    alcohol = gr.Number(
                        label="Alcohol (%)",
                        value=13.0,
                        minimum=11.0,
                        maximum=15.0
                    )
                    
                    acido_malico = gr.Number(
                        label="Ácido Málico (g/L)",
                        value=2.0,
                        minimum=0.5,
                        maximum=6.0
                    )
                    
                    ceniza = gr.Number(
                        label="Ceniza (g/L)",
                        value=2.4,
                        minimum=1.0,
                        maximum=4.0
                    )
                    
                    alcalinidad_ceniza = gr.Number(
                        label="Alcalinidad de la Ceniza",
                        value=20.0,
                        minimum=10.0,
                        maximum=30.0
                    )
                    
                    magnesio = gr.Number(
                        label="Magnesio (mg/L)",
                        value=100.0,
                        minimum=70.0,
                        maximum=162.0
                    )
                
                with gr.Column():
                    gr.Markdown("### Características Fenólicas")
                    
                    fenoles_totales = gr.Number(
                        label="Fenoles Totales",
                        value=2.5,
                        minimum=0.5,
                        maximum=4.0
                    )
                    
                    flavonoides = gr.Number(
                        label="Flavonoides",
                        value=2.0,
                        minimum=0.0,
                        maximum=5.0
                    )
                    
                    fenoles_no_flavonoides = gr.Number(
                        label="Fenoles No Flavonoides",
                        value=0.3,
                        minimum=0.1,
                        maximum=0.7
                    )
                    
                    proantocianinas = gr.Number(
                        label="Proantocianinas",
                        value=1.5,
                        minimum=0.4,
                        maximum=4.0
                    )
                
                with gr.Column():
                    gr.Markdown("### Características del Color")
                    
                    intensidad_color = gr.Number(
                        label="Intensidad del Color",
                        value=5.0,
                        minimum=1.0,
                        maximum=13.0
                    )
                    
                    matiz = gr.Number(
                        label="Matiz",
                        value=1.0,
                        minimum=0.5,
                        maximum=1.7
                    )
                    
                    od280_od315 = gr.Number(
                        label="OD280/OD315 de Vinos Diluidos",
                        value=3.0,
                        minimum=1.3,
                        maximum=4.0
                    )
                    
                    prolina = gr.Number(
                        label="Prolina (mg/L)",
                        value=1000.0,
                        minimum=278.0,
                        maximum=1680.0
                    )
            
            predecir_btn = gr.Button(
                "Clasificar Vino",
                variant="primary",
                size="lg"
            )
            
            resultado = gr.Markdown(label="Resultado")
            
            predecir_btn.click(
                fn=self.realizar_prediccion,
                inputs=[
                    alcohol, acido_malico, ceniza, alcalinidad_ceniza,
                    magnesio, fenoles_totales, flavonoides,
                    fenoles_no_flavonoides, proantocianinas,
                    intensidad_color, matiz, od280_od315, prolina
                ],
                outputs=[resultado]
            )
            
            gr.Markdown(
                """
                ### Información del Modelo
                
                Este clasificador utiliza 13 características químicas para 
                predecir la variedad del vino. Los valores sugeridos son 
                rangos típicos basados en el dataset Wine.
                """
            )
        
        return interfaz