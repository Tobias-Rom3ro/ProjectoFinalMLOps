import logging
import gradio as gr
from typing import List, Tuple
from app.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class LLMInterface:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.historial_chat: List[Tuple[str, str]] = []
    
    def procesar_mensaje(
        self, 
        mensaje: str, 
        historial: List[Tuple[str, str]]
    ) -> Tuple[List[Tuple[str, str]], str]:
        if not mensaje or mensaje.strip() == "":
            return historial, ""
        
        try:
            contexto = self._construir_contexto(historial)
            
            respuesta = self.llm_client.consultar(
                pregunta=mensaje,
                contexto=contexto
            )
            
            historial.append((mensaje, respuesta))
            
            return historial, ""
        
        except Exception as error:
            logger.error(f"Error al procesar mensaje: {error}")
            error_msg = f"Error: No se pudo procesar el mensaje. {str(error)}"
            historial.append((mensaje, error_msg))
            return historial, ""
    
    def _construir_contexto(
        self, 
        historial: List[Tuple[str, str]]
    ) -> str:
        if len(historial) < 2:
            return ""
        
        contexto_partes = []
        for pregunta, respuesta in historial[-3:]:
            contexto_partes.append(f"Usuario: {pregunta}")
            contexto_partes.append(f"Asistente: {respuesta}")
        
        return "\n".join(contexto_partes)
    
    def limpiar_historial(self) -> List[Tuple[str, str]]:
        logger.info("Historial de chat limpiado")
        return []
    
    def crear_interfaz(self) -> gr.Blocks:
        with gr.Blocks() as interfaz:
            gr.Markdown(
                """
                # Asistente de Conversación LLM
                
                Interactúa con el modelo de lenguaje Gemini para obtener 
                respuestas a tus preguntas. El asistente mantiene el 
                contexto de la conversación.
                """
            )
            
            chatbot = gr.Chatbot(
                label="Conversación",
                height=400,
                type="messages"
            )
            
            with gr.Row():
                mensaje_input = gr.Textbox(
                    label="Tu mensaje",
                    placeholder="Escribe tu pregunta aquí...",
                    lines=2,
                    scale=4
                )
                enviar_btn = gr.Button("Enviar", scale=1, variant="primary")
            
            limpiar_btn = gr.Button("Limpiar conversación")
            
            estado_salud = gr.Textbox(
                label="Estado del servicio",
                interactive=False
            )
            
            def actualizar_estado():
                try:
                    salud = self.llm_client.verificar_salud()
                    return f"Estado: {salud.get('status', 'desconocido')}"
                except:
                    return "Estado: No disponible"
            
            verificar_btn = gr.Button("Verificar estado del servicio")
            
            enviar_btn.click(
                fn=self.procesar_mensaje,
                inputs=[mensaje_input, chatbot],
                outputs=[chatbot, mensaje_input]
            )
            
            mensaje_input.submit(
                fn=self.procesar_mensaje,
                inputs=[mensaje_input, chatbot],
                outputs=[chatbot, mensaje_input]
            )
            
            limpiar_btn.click(
                fn=self.limpiar_historial,
                outputs=[chatbot]
            )
            
            verificar_btn.click(
                fn=actualizar_estado,
                outputs=[estado_salud]
            )
        
        return interfaz