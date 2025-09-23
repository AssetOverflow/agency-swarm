from agents import OpenAIChatCompletionsModel
from agents.models.interface import Model, ModelProvider
from openai import AsyncOpenAI
import os


class GroqModel(OpenAIChatCompletionsModel):
    """Groq model implementation using OpenAI-compatible API."""

    def __init__(self, model_name: str, api_key: str = None, **kwargs):
        if api_key is None:
            api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is required for Groq models")
        
        # Create OpenAI client with Groq base URL
        client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )
        
        super().__init__(model=model_name, openai_client=client)


class GroqProvider(ModelProvider):
    """Provider for Groq models using OpenAI-compatible API."""

    def get_model(self, model_name: str = "mixtral-8x7b-32768", api_key: str = None, **kwargs) -> Model:
        return GroqModel(model_name, api_key=api_key, **kwargs)