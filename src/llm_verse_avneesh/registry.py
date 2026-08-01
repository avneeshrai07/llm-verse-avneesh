from __future__ import annotations
import logging
from collections.abc import Callable
from typing import Any

from ._register import _REGISTRY          # ← shared dict, no circular dep
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
