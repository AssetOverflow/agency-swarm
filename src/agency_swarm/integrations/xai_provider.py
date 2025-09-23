from agents.models.interface import Model, ModelProvider
from xai_sdk import Client  # or AsyncClient if required
import os
from typing import AsyncIterator

class XAIModel(Model):
    def __init__(self, model_name: str, api_key: str = None, **kwargs):
        if api_key is None:
            api_key = os.getenv("XAI_API_KEY")
        if not api_key:
            raise ValueError("XAI_API_KEY environment variable is required for XAI models")
        self.client = Client(api_key=api_key)
        self.model_name = model_name

    def get_response(self, system_instructions, input, *args, **kwargs):
        # Adapt message packaging as needed by agency-swarm
        from xai_sdk.chat import system, user
        messages = []
        if system_instructions:
            messages.append(system(system_instructions))
        messages.append(user(input))
        chat = self.client.chat.create(model=self.model_name, messages=messages)
        response = chat.sample()   # Use .stream() for streaming
        return response.content    # Adapt this if agency-swarm expects richer objects

    async def stream_response(self, system_instructions, input, *args, **kwargs) -> AsyncIterator:
        # For now, raise NotImplementedError as streaming is not implemented
        raise NotImplementedError("Streaming is not yet implemented for XAI provider")

class XAIProvider(ModelProvider):
    def get_model(self, model_name: str = "grok-4", api_key: str = None, **kwargs) -> Model:
        return XAIModel(model_name, api_key=api_key)
