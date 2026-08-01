from __future__ import annotations

import pytest

import llm_verse_avneesh  # noqa: F401  -- triggers provider registration
from llm_verse_avneesh.registry import get_handler, is_registered, list_providers
from llm_verse_avneesh.exceptions import ProviderNotFoundError


EXPECTED_PROVIDERS = {
    "claude-haiku-4-5",
    "claude-haiku-4-5-agentic",
    "nova-lite",
    "nova-2-lite",
    "nova-2-lite-grounding",
    "nova-pro",
    "gemini-3.1-flash-lite",
}


def test_all_expected_providers_registered():
    registered = set(list_providers())
    assert EXPECTED_PROVIDERS.issubset(registered)


def test_is_registered_true_for_known_provider():
    assert is_registered("nova-lite")


def test_is_registered_false_for_unknown_provider():
    assert not is_registered("does-not-exist")


def test_get_handler_raises_for_unknown_provider():
    with pytest.raises(ProviderNotFoundError):
        get_handler("does-not-exist")


def test_get_handler_returns_callable_for_known_provider():
    handler = get_handler("nova-lite")
    assert callable(handler)


def test_dead_temp_provider_removed():
    # amazon_nova2_lite_temp was a dead experimental duplicate of nova-2-lite
    # in the original repo; it should not exist under any name here.
    assert "amazon_nova2_lite_temp" not in list_providers()
