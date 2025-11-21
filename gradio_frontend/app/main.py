# file: gradio_frontend/app/main.py
import logging
import gradio as gr

from app.core.config import get_settings
from app.core.logging_config import setup_logging
from app.services.llm_client import LLMClient
from app.services.sklearn_client import SklearnClient
from app.services.cnn_client import CNNClient
from app.ui.llm_interface import LLMInterface
from app.ui.sklearn_interface import SklearnInterface
from app.ui.cnn_interface import CNNInterface


settings = get_settings()
setup_logging(settings.service_name, settings.log_level)
logger = logging.getLogger(__name__)


def crear_aplicacion() -> gr.Blocks:
    logger.info("Inicializando aplicación Gradio")
    
    llm_client = LLMClient(
        base_url=settings.llm_connector_url,
        timeout=settings.request_timeout
    )
    
    sklearn_client = SklearnClient(
        base_url=settings.sklearn_model_url,
        timeout=settings.request_timeout
    )
    
    cnn_client = CNNClient(
        base_url=settings.cnn_image_url,
        timeout=settings.request_timeout
    )
    
    llm_interface = LLMInterface(llm_client)
    sklearn_interface = SklearnInterface(sklearn_client)
    cnn_interface = CNNInterface(cnn_client)
    
    with gr.Blocks(
        title="Pipeline Inteligente MLOps",
        theme=gr.themes.Soft()
    ) as app:
        gr.Markdown(
            """
            # Pipeline Inteligente: LLM + ML Clásico + CNN
            
            Sistema integrado de Machine Learning que combina tres 
            servicios independientes:
            
            - **LLM:** Asistente conversacional con Gemini
            - **ML Clásico:** Clasificador de vinos con Random Forest
            - **CNN:** Clasificador de dígitos con Red Neuronal Convolucional
            Selecciona una pestaña para interactuar con cada servicio.
            """
        )
        
        with gr.Tabs():
            with gr.Tab("Asistente LLM"):
                llm_interface.crear_interfaz()
            
            with gr.Tab("Clasificador de Vinos"):
                sklearn_interface.crear_interfaz()
            
            with gr.Tab("Clasificador de Imágenes"):
                cnn_interface.crear_interfaz()
            
            with gr.Tab("Estado del Sistema"):
                gr.Markdown("## Estado de los Servicios")
                
                with gr.Row():
                    with gr.Column():
                        estado_llm = gr.Textbox(
                            label="Servicio LLM",
                            interactive=False
                        )
                        verificar_llm_btn = gr.Button("Verificar LLM")
                    
                    with gr.Column():
                        estado_sklearn = gr.Textbox(
                            label="Servicio Sklearn",
                            interactive=False
                        )
                        verificar_sklearn_btn = gr.Button("Verificar Sklearn")
                    
                    with gr.Column():
                        estado_cnn = gr.Textbox(
                            label="Servicio CNN",
                            interactive=False
                        )
                        verificar_cnn_btn = gr.Button("Verificar CNN")
                
                def verificar_llm():
                    try:
                        salud = llm_client.verificar_salud()
                        return f"Estado: {salud.get('status', 'desconocido')}"
                    except Exception as e:
                        return f"Error: {str(e)}"
                
                def verificar_sklearn():
                    try:
                        salud = sklearn_client.verificar_salud()
                        return f"Estado: {salud.get('estado', 'desconocido')}"
                    except Exception as e:
                        return f"Error: {str(e)}"
                
                def verificar_cnn():
                    try:
                        salud = cnn_client.verificar_salud()
                        return f"Estado: {salud.get('status', 'desconocido')}"
                    except Exception as e:
                        return f"Error: {str(e)}"
                
                verificar_llm_btn.click(
                    fn=verificar_llm,
                    outputs=[estado_llm]
                )
                
                verificar_sklearn_btn.click(
                    fn=verificar_sklearn,
                    outputs=[estado_sklearn]
                )
                
                verificar_cnn_btn.click(
                    fn=verificar_cnn,
                    outputs=[estado_cnn]
                )
                
                gr.Markdown(
                    """
                    ### Información del Sistema
                    
                    Este sistema está compuesto por servicios independientes 
                    que se comunican mediante APIs REST. Cada servicio puede 
                    verificarse de manera individual.
                    
                    **Arquitectura:**
                    - Contenedores Docker independientes
                    - Registro con MLflow
                    - Logging estructurado en JSON
                    - CI/CD con GitHub Actions
                    """
                )
    
    logger.info("Aplicación Gradio creada exitosamente")
    return app


def main():
    logger.info(f"Iniciando servidor Gradio en {settings.host}:{settings.port}")
    
    app = crear_aplicacion()
    
    app.launch(
        server_name=settings.host,
        server_port=settings.port,
        share=False
    )


if __name__ == "__main__":
    main()