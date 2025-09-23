"""
Utility functions for provider selection and configuration.
"""

import os
from typing import Optional

from .groq_provider import GroqProvider
from .xai_provider import XAIProvider


def get_available_providers():
    """
    Get a list of available providers based on environment variables.

    Returns:
        dict: Dictionary mapping provider names to provider classes
    """
    providers = {}

    if os.getenv("XAI_API_KEY"):
        providers["xai"] = XAIProvider

    if os.getenv("GROQ_API_KEY"):
        providers["groq"] = GroqProvider

    return providers


def select_provider(provider_name: Optional[str] = None):
    """
    Select a provider automatically or by name.

    Args:
        provider_name: Name of the provider to select ('xai', 'groq', etc.)
                      If None, selects the first available provider.

    Returns:
        ModelProvider: The selected provider class

    Raises:
        ValueError: If no providers are available or specified provider is not found
    """
    available = get_available_providers()

    if not available:
        raise ValueError("No API keys found for any providers. Please set XAI_API_KEY or GROQ_API_KEY.")

    if provider_name:
        if provider_name not in available:
            raise ValueError(f"Provider '{provider_name}' not available. Available providers: {list(available.keys())}")
        return available[provider_name]

    # Return the first available provider
    return list(available.values())[0]


def get_model_from_provider(provider_name: Optional[str] = None, model_name: Optional[str] = None, **kwargs):
    """
    Convenience function to get a model from an available provider.

    Args:
        provider_name: Name of the provider ('xai', 'groq', etc.). If None, uses first available.
        model_name: Name of the model. If None, uses provider default.
        **kwargs: Additional arguments passed to get_model()

    Returns:
        Model: The model instance

    Raises:
        ValueError: If no providers are available
    """
    provider_class = select_provider(provider_name)
    provider = provider_class()

    if model_name:
        return provider.get_model(model_name, **kwargs)
    else:
        return provider.get_model(**kwargs)