import logging
from typing import Optional
from google import genai


logger = logging.getLogger(__name__)


class GeminiService:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        try:
            self.client = genai.Client(api_key=self.api_key)
            logger.info("Gemini client initialized successfully")
        except Exception as error:
            logger.error(f"Failed to initialize Gemini client: {error}")
            raise RuntimeError(f"Could not initialize Gemini client: {error}")
    
    def generate_response(
        self, 
        prompt: str, 
        context: Optional[str] = None,
        model: str = "gemini-2.0-flash-exp"
    ) -> str:
        if not self.client:
            logger.error("Gemini client is not initialized")
            raise RuntimeError("Gemini client is not available")
        
        try:
            full_prompt = self._build_prompt(prompt, context)
            
            logger.info(
                f"Generating content with model: {model}, "
                f"prompt length: {len(prompt)}"
            )
            
            response = self.client.models.generate_content(
                model=model,
                contents=full_prompt
            )
            
            logger.info("Content generated successfully")
            return response.text
        
        except Exception as error:
            logger.error(f"Error generating content: {error}")
            raise RuntimeError(f"Failed to generate content: {error}")
    
    def _build_prompt(self, prompt: str, context: Optional[str]) -> str:
        if context:
            return f"Context: {context}\n\nQuery: {prompt}"
        return prompt
    
    def is_available(self) -> bool:
        return self.client is not None