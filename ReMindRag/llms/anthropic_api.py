from .base import AgentBase
from anthropic import Anthropic
from typing import Optional, List, Dict, Any

class AnthropicAgent(AgentBase):
    def __init__(self, api_key: str, llm_model_name: str, max_retries: int = 3):
        self.llm_model_name = llm_model_name
        self.api_key = api_key
        self.max_retries = max_retries

        self.client = Anthropic(
            api_key=self.api_key,
            max_retries=self.max_retries
        )
        
    def generate_response(self, system_prompt: Optional[str], chat_history: List[Dict[str, Any]]) -> str:
        # Anthropic doesn't allow system messages in the normal messages array. 
        # It takes it as a separate keyword argument.
        
        # Convert the OpenAI-style chat history to Anthropic-style if needed
        messages = []
        for msg in chat_history:
            # Anthropic roles are strictly 'user' or 'assistant'
            role = msg.get("role")
            if role == "system":
                continue # Skip any lingering system prompts in history
            elif role not in ["user", "assistant"]:
                role = "user" # Default to user for safety
            
            messages.append({
                "role": role,
                "content": msg.get("content", "")
            })

        for attempt in range(self.max_retries + 1):
            try:
                kwargs = {
                    "model": self.llm_model_name,
                    "max_tokens": 4096,
                    "temperature": 0.0,
                    "messages": messages
                }
                
                if system_prompt:
                    kwargs["system"] = system_prompt

                response = self.client.messages.create(**kwargs)
                
                # Anthropic's response structure is slightly different
                return response.content[0].text
                
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    continue
                else:
                    raise Exception(f"Failed after {self.max_retries} retries. Last error: {str(last_error)}") from last_error
