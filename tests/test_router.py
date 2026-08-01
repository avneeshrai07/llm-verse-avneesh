from __future__ import annotations

import pytest

from llm_verse_avneesh.router import Router
from llm_verse_avneesh.exceptions import RouterValidationError, ProviderNotFoundError
from tests.conftest import aws_credentials


@pytest.fixture
def router():
    return Router()


async def test_successful_dispatch_returns_dict(router, fake_provider):
    result = await router.get_response(
        llm_name=fake_provider,
        system_prompt="be helpful",
        user_prompt="hi",
        context=None,
        temperature=0.5,
        pydantic_model=None,
        max_tokens=100,
        repo_name="repo",
        llm_identifier="id-1",
        **aws_credentials(),
    )
    assert isinstance(result, dict)
    assert result["response"] == "echo: hi"


async def test_unregistered_provider_raises_provider_not_found(router):
    with pytest.raises(ProviderNotFoundError):
        await router.get_response(
            llm_name="totally-unregistered-model",
            system_prompt="x",
            user_prompt="y",
            context=None,
            temperature=0.5,
            pydantic_model=None,
            max_tokens=100,
            repo_name="repo",
            llm_identifier="id-1",
            **aws_credentials(),
        )


async def test_invalid_input_raises_router_validation_error(router):
    with pytest.raises(RouterValidationError):
        await router.get_response(
            llm_name="nova-lite",
            system_prompt="",  # empty -> fails LLMRequest validation
            user_prompt="y",
            context=None,
            temperature=0.5,
            pydantic_model=None,
            max_tokens=100,
            repo_name="repo",
            llm_identifier="id-1",
            **aws_credentials(),
        )


async def test_temperature_over_model_limit_raises_router_validation_error(router):
    # nova-lite's limit is temperature_max=1.0 (see limits.py)
    with pytest.raises(RouterValidationError, match="temperature"):
        await router.get_response(
            llm_name="nova-lite",
            system_prompt="be helpful",
            user_prompt="hi",
            context=None,
            temperature=1.5,
            pydantic_model=None,
            max_tokens=100,
            repo_name="repo",
            llm_identifier="id-1",
            **aws_credentials(),
        )


async def test_max_tokens_over_model_limit_raises_router_validation_error(router):
    # nova-lite's limit is max_output_tokens=5000 (see limits.py)
    with pytest.raises(RouterValidationError, match="max_tokens"):
        await router.get_response(
            llm_name="nova-lite",
            system_prompt="be helpful",
            user_prompt="hi",
            context=None,
            temperature=0.5,
            pydantic_model=None,
            max_tokens=999999,
            repo_name="repo",
            llm_identifier="id-1",
            **aws_credentials(),
        )


async def test_gemini_temperature_ceiling_is_higher_than_bedrock(router, monkeypatch):
    # gemini-3.1-flash-lite allows temperature up to 2.0 — this must not be
    # rejected by a one-size-fits-all cap the way the old blanket 0-1 range did.
    from llm_verse_avneesh import limits as limits_module

    called = {}

    async def fake_handler(request):
        called["temperature"] = request.temperature
        return {
            "response": "ok",
            "provider": "google",
            "model": "gemini-3.1-flash-lite",
            "input_tokens": 1,
            "output_tokens": 1,
            "llm_identifier": request.llm_identifier,
        }

    from llm_verse_avneesh._register import _REGISTRY
    original = _REGISTRY.get("gemini-3.1-flash-lite")
    _REGISTRY["gemini-3.1-flash-lite"] = fake_handler
    try:
        result = await router.get_response(
            llm_name="gemini-3.1-flash-lite",
            system_prompt="be helpful",
            user_prompt="hi",
            context=None,
            temperature=1.8,
            pydantic_model=None,
            max_tokens=100,
            repo_name="repo",
            llm_identifier="id-1",
            google_api_key="fake-google-key",
        )
    finally:
        if original is not None:
            _REGISTRY["gemini-3.1-flash-lite"] = original

    assert called["temperature"] == 1.8
    assert result["response"] == "ok"
