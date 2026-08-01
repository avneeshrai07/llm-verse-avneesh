from __future__ import annotations

import pytest

from llm_verse_avneesh._register import register, _REGISTRY

FAKE_MODEL_NAME = "test-fake-model"


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


def aws_credentials():
    return dict(
        region_name="us-east-1",
        aws_access_key_id="AKIAFAKEFAKEFAKEFAKE",
        aws_secret_access_key="fake-secret-value-not-real",
    )
