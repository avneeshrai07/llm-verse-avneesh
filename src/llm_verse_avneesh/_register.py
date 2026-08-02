from __future__ import annotations
from collections.abc import Callable
from typing import Any

_REGISTRY: dict[str, Callable[..., Any]] = {}

def register(name: str):
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        _REGISTRY[name] = fn
        return fn
    return decorator


# Separate from _REGISTRY: embedding models take a fundamentally different
# request shape (text in, vector out - no system_prompt/temperature/
# pydantic_model/tools) than the chat/generation models above, so they get
# their own namespace rather than sharing one registry keyed by name.
_EMBEDDING_REGISTRY: dict[str, Callable[..., Any]] = {}

def register_embedding(name: str):
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        _EMBEDDING_REGISTRY[name] = fn
        return fn
    return decorator
