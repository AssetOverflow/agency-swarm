# Integrations

This directory contains custom model adapters for external LLM sources.
- `xai_provider.py`: Direct integration with xAI's Grok models (xai-sdk; Official Python 3.10+ SDK).
- `groq_provider.py`: Convenience provider for Groq models (OpenAI-compatible API).
- `provider_utils.py`: Utility functions for automatic provider selection based on available API keys.
- `openai_model.py` or via `LitellmModel`: OpenAI API and compatible endpoints (e.g., Groq).

## Example: Using Grok (xAI) Directly

    ```python

    from agency_swarm.integrations.xai_provider import XAIProvider
    agent = Agent(model=XAIProvider().get_model("grok-4-fast-reasoning"))

    ```

## Example: Using Groq

    ```python

    from agency_swarm.integrations.groq_provider import GroqProvider
    agent = Agent(model=GroqProvider().get_model("mixtral-8x7b-32768"))

    ```

## Example: Automatic Provider Selection

    ```python

    from agency_swarm.integrations.provider_utils import get_model_from_provider
    agent = Agent(model=get_model_from_provider())  # Uses first available provider

    ```