import logging
import gradio as gr
from typing import List, Tuple
from app.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class LLMInterface:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
    
    def procesar_mensaje(
        self, 
        mensaje: str, 
        historial: List
    ) -> Tuple[List, str]:
        """Procesa mensaje y devuelve historial actualizado."""
        if not mensaje or mensaje.strip() == "":
            return historial, ""
        
        try:
            # Construir contexto desde el historial
            contexto = self._construir_contexto_desde_historial(historial)
            
            # Obtener respuesta del LLM
            respuesta = self.llm_client.consultar(
                pregunta=mensaje,
                contexto=contexto
            )
            
            # Agregar al historial en formato correcto
            historial.append({"role": "user", "content": mensaje})
            historial.append({"role": "assistant", "content": respuesta})
            
            return historial, ""
        
        except Exception as error:
            logger.error(f"Error al procesar mensaje: {error}")
            error_msg = f"Error: No se pudo procesar el mensaje. {str(error)}"
            historial.append({"role": "user", "content": mensaje})
            historial.append({"role": "assistant", "content": error_msg})
            return historial, ""
    
    def _construir_contexto_desde_historial(self, historial: List) -> str:
        """Construye contexto textual desde el historial."""
        if not historial or len(historial) < 2:
            return ""
        
        contexto_partes = []
        # Tomar últimos 6 mensajes (3 intercambios)
        for msg in historial[-6:]:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                contexto_partes.append(f"Usuario: {content}")
            elif role == "assistant":
                contexto_partes.append(f"Asistente: {content}")
        
        return "\n".join(contexto_partes)
    
    def limpiar_historial(self) -> List:
        """Limpia el historial de chat."""
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
            
            # Cambiar type a None (default) para compatibilidad
            chatbot = gr.Chatbot(
                label="Conversación",
                height=400
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
                except Exception as e:
                    return f"Estado: Error - {str(e)}"
            
            verificar_btn = gr.Button("Verificar estado del servicio")
            
            # Conectar eventos
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