from __future__ import annotations
import logging
from collections.abc import Callable
from typing import Any

from ._register import _REGISTRY, _EMBEDDING_REGISTRY   # ← shared dicts, no circular dep
from .exceptions import ProviderNotFoundError

logger = logging.getLogger(__name__)


def get_handler(llm_name: str) -> Callable[..., Any]:
    name = llm_name.strip()
    if name not in _REGISTRY:
        raise ProviderNotFoundError(
            f"No LLM registered as {name!r}. Available: {list_providers()}"
        )
    return _REGISTRY[name]


def list_providers() -> list[str]:
    return sorted(_REGISTRY.keys())


def is_registered(llm_name: str) -> bool:
    return llm_name.strip() in _REGISTRY


def get_embedding_handler(llm_name: str) -> Callable[..., Any]:
    name = llm_name.strip()
    if name not in _EMBEDDING_REGISTRY:
        raise ProviderNotFoundError(
            f"No embedding model registered as {name!r}. Available: {list_embedding_providers()}"
        )
    return _EMBEDDING_REGISTRY[name]


def list_embedding_providers() -> list[str]:
    return sorted(_EMBEDDING_REGISTRY.keys())


def is_embedding_registered(llm_name: str) -> bool:
    return llm_name.strip() in _EMBEDDING_REGISTRY
