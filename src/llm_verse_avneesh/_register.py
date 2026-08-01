from __future__ import annotations
from collections.abc import Callable
from typing import Any

_REGISTRY: dict[str, Callable[..., Any]] = {}

def register(name: str):
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        _REGISTRY[name] = fn
        return fn
    return decorator
