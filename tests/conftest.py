from __future__ import annotations

import pytest

from llm_verse_avneesh._register import register, register_embedding, _REGISTRY, _EMBEDDING_REGISTRY

FAKE_MODEL_NAME = "test-fake-model"
FAKE_EMBEDDING_MODEL_NAME = "test-fake-embedding-model"


@pytest.fixture
def fake_provider():
    """Register a network-free fake provider for router/dispatch tests."""

    @register(FAKE_MODEL_NAME)
    async def _fake_handler(request):
        return {
            "response": f"echo: {request.user_prompt}",
            "provider": "fake",
            "model": FAKE_MODEL_NAME,
            "input_tokens": 1,
            "output_tokens": 1,
            "llm_identifier": request.llm_identifier,
            "latency_ms": 0,
        }

    yield FAKE_MODEL_NAME
    _REGISTRY.pop(FAKE_MODEL_NAME, None)


@pytest.fixture
def fake_embedding_provider():
    """Register a network-free fake embedding provider for router/dispatch tests."""

    @register_embedding(FAKE_EMBEDDING_MODEL_NAME)
    async def _fake_embedding_handler(request):
        # Deterministic, text-length-derived "vector" -- no real model call.
        vector = [float(len(request.text))] * (request.dimensions or 4)
        return {
            "embedding": vector,
            "dimensions": len(vector),
            "provider": "fake",
            "model": FAKE_EMBEDDING_MODEL_NAME,
            "input_tokens": 0,
            "llm_identifier": request.llm_identifier,
            "latency_ms": 0,
        }

    yield FAKE_EMBEDDING_MODEL_NAME
    _EMBEDDING_REGISTRY.pop(FAKE_EMBEDDING_MODEL_NAME, None)


def aws_credentials():
    return dict(
        region_name="us-east-1",
        aws_access_key_id="AKIAFAKEFAKEFAKEFAKE",
        aws_secret_access_key="fake-secret-value-not-real",
    )
