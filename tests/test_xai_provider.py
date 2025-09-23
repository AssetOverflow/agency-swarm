# tests/test_xai_provider.py

import os
import pytest

from agency_swarm.integrations.xai_provider import XAIProvider
from agents import ModelSettings, ModelTracing

@pytest.mark.skipif(
    "XAI_API_KEY" not in os.environ,
    reason="Set XAI_API_KEY in your environment to test xAI integration."
)
def test_grok_model_basic_response():
    model = XAIProvider().get_model("grok-4")
    system_prompt = "You are a helpful assistant."
    user_message = "What is the capital of France?"
    
    # Use proper signature for get_response
    result = model.get_response(
        system_instructions=system_prompt,
        input=user_message,
        model_settings=ModelSettings(),
        tools=[],
        output_schema=None,
        handoffs=[],
        tracing=ModelTracing.DISABLED,
        previous_response_id=None
    )
    assert isinstance(result, str)  # XAI returns string directly
    assert "paris" in result.lower()  # Grok should return Paris as capital

@pytest.mark.skipif(
    "XAI_API_KEY" not in os.environ,
    reason="Set XAI_API_KEY in your environment to test xAI integration."
)
def test_grok_with_reasoning():
    model = XAIProvider().get_model("grok-4")
    system_prompt = "You are a math tutor. Respond with reasoning."
    user_message = "What is 17+25?"
    
    # Use proper signature for get_response
    result = model.get_response(
        system_instructions=system_prompt,
        input=user_message,
        model_settings=ModelSettings(),
        tools=[],
        output_schema=None,
        handoffs=[],
        tracing=ModelTracing.DISABLED,
        previous_response_id=None
    )
    assert isinstance(result, str)  # XAI returns string directly
    assert any(char.isdigit() for char in result)  # Should include '42'
