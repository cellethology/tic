"""
This module provides different LLM backends for annotation.
"""
from __future__ import annotations

from .openai import OpenAIChatGPT

__all__ = ["OpenAIChatGPT", "get_llm"]


def get_llm(model_name: str, model_kwargs: dict = None):
    """
    Factory to retrieve the appropriate LLM backend.
    """
    model_kwargs = model_kwargs or {}
    if model_name == "openai":
        return OpenAIChatGPT(**model_kwargs)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")