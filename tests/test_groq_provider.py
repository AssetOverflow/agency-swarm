# tests/test_groq_provider.py

import os
import pytest

from agency_swarm.integrations.groq_provider import GroqProvider
from agents import ModelSettings, ModelTracing

@pytest.mark.skipif(
    "GROQ_API_KEY" not in os.environ,
    reason="Set GROQ_API_KEY in your environment to test Groq integration."
)
@pytest.mark.asyncio
async def test_groq_model_basic_response():
    model = GroqProvider().get_model("llama-3.1-8b-instant")
    system_prompt = "You are a helpful assistant."
    user_message = "What is the capital of France?"
    
    # Use proper signature for get_response
    result = await model.get_response(
        system_instructions=system_prompt,
        input=user_message,
        model_settings=ModelSettings(),
        tools=[],
        output_schema=None,
        handoffs=[],
        tracing=ModelTracing.DISABLED,
        previous_response_id=None
    )
    assert isinstance(result, object)  # ModelResponse object
    assert hasattr(result, 'output')  # Should have output attribute

@pytest.mark.skipif(
    "GROQ_API_KEY" not in os.environ,
    reason="Set GROQ_API_KEY in your environment to test Groq integration."
)
@pytest.mark.asyncio
async def test_groq_with_custom_model():
    model = GroqProvider().get_model("openai/gpt-oss-20b")
    system_prompt = "You are a helpful assistant."
    user_message = "Hello, world!"
    
    # Use proper signature for get_response
    result = await model.get_response(
        system_instructions=system_prompt,
        input=user_message,
        model_settings=ModelSettings(),
        tools=[],
        output_schema=None,
        handoffs=[],
        tracing=ModelTracing.DISABLED,
        previous_response_id=None
    )
    assert isinstance(result, object)  # ModelResponse object
    assert hasattr(result, 'output')  # Should have output attribute