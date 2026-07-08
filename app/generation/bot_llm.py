import re
from abc import ABC, abstractmethod

class LLMClient(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        ...
    
class StubLLMClient(LLMClient):
    """
    Offline default: not network call, no API key required. It doesn NOT synthesize answers - it returns the top retrieved passage.
    """
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        match = re.search(r"\[1\][^\n]*\n(.*?)(?=\n\n[2\]|\n\nQuestion:)", user_prompt, re.S)
        snippet = match.group(1).strip() if match else "no context retrieved"
        return (
            f"{snippet} [1]\n\n"
            "[StubLLMClient: no LLM backend configured - this the top retrieved passage]"
        )
        
class AnthropicLLMClient(LLMClient):
    """Call Anthropic API directly over HTTPS"""
    
    def __init__(self, api_key:str, model:str, max_tokens: int = 1024):
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        import requests
        
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "system": system_prompt,
                "messages": [{"role": "user", "content":user_prompt}],
            },
            timeout= 30,
        )
        response.raise_for_status()
        data = response.json()
        return "".join(block.get("text","") for block in data.get("content",[]) if block.get("type") == "text")

def get_llm_client() -> LLMClient:
    from app.config import settings
    
    if settings.llm_backend == "anthropic":
        if not settings.anthropic_api_key:
            raise RuntimeError("requires the API key environment variable to be set")
        return AnthropicLLMClient(
            api_key=settings.anthropic_api_key,
            model = settings.llm_model,
            max_tokens=settings.llm_max_tokens,
        )
    return StubLLMClient()
        