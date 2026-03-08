from .base import AgentBase
from google import genai
from google.genai import types
from typing import Optional, List, Dict, Any

class GeminiAgent(AgentBase):
    def __init__(self, api_key: str, llm_model_name: str, max_retries: int = 3):
        self.llm_model_name = llm_model_name
        self.api_key = api_key
        self.max_retries = max_retries

        self.client = genai.Client(api_key=self.api_key)
        
    def generate_response(self, system_prompt: Optional[str], chat_history: List[Dict[str, Any]]) -> str:
        # Convert OpenAI-style chat history to Gemini's format
        contents = []
        for msg in chat_history:
            role = msg.get("role")
            if role == "system":
                continue # Handled via system_instruction
                
            # Map roles: 'assistant' -> 'model', everything else -> 'user'
            gemini_role = "model" if role == "assistant" else "user"
            contents.append(
                types.Content(
                    role=gemini_role,
                    parts=[types.Part.from_text(text=msg.get("content", ""))]
                )
            )

        kwargs = {
            "model": self.llm_model_name,
            "contents": contents,
            "config": types.GenerateContentConfig(
                temperature=0.0,
            )
        }
        
        if system_prompt:
            kwargs["config"].system_instruction = system_prompt

        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.models.generate_content(**kwargs)
                return response.text
                
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    continue
                else:
                    raise Exception(f"Failed after {self.max_retries} retries. Last error: {str(last_error)}") from last_error
