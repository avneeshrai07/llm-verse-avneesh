from __future__ import annotations

import pytest
from pydantic import ValidationError

from llm_verse_avneesh.models import EmbeddingRequest
from llm_verse_avneesh.types import EmbeddingResponse
from llm_verse_avneesh.router import Router
from llm_verse_avneesh.registry import (
    get_embedding_handler,
    list_embedding_providers,
    is_embedding_registered,
)
from llm_verse_avneesh.exceptions import RouterValidationError, ProviderNotFoundError
from llm_verse_avneesh.help import list_embedding_models, embedding_model_info
from tests.conftest import aws_credentials


def make_embedding_request(**overrides):
    base = dict(
        llm_name="titan-embed-v2",
        text="hello world",
        dimensions=None,
        repo_name="repo",
        llm_identifier="id-1",
        **aws_credentials(),
    )
    base.update(overrides)
    return EmbeddingRequest(**base)


# ── EmbeddingRequest validation ─────────────────────────────────────────

def test_valid_embedding_request_builds():
    req = make_embedding_request()
    assert req.llm_name == "titan-embed-v2"
    assert req.text == "hello world"


def test_empty_text_rejected():
    with pytest.raises(ValidationError):
        make_embedding_request(text="   ")


def test_negative_dimensions_rejected():
    with pytest.raises(ValidationError):
        make_embedding_request(dimensions=-1)


def test_zero_dimensions_rejected():
    with pytest.raises(ValidationError):
        make_embedding_request(dimensions=0)


def test_missing_aws_credentials_rejected():
    with pytest.raises(ValidationError):
        EmbeddingRequest(
            llm_name="titan-embed-v2", text="hi",
            repo_name="repo", llm_identifier="id-1",
        )


# ── EmbeddingResponse validation ────────────────────────────────────────

def test_valid_embedding_response_builds():
    resp = EmbeddingResponse(
        embedding=[0.1, 0.2, 0.3], dimensions=3,
        provider="bedrock", model="amazon.titan-embed-text-v2:0",
        llm_identifier="id-1",
    )
    assert len(resp.embedding) == 3


def test_empty_embedding_rejected():
    with pytest.raises(ValidationError):
        EmbeddingResponse(
            embedding=[], dimensions=0,
            provider="bedrock", model="x", llm_identifier="id-1",
        )


def test_mismatched_dimensions_rejected():
    with pytest.raises(ValidationError):
        EmbeddingResponse(
            embedding=[0.1, 0.2, 0.3], dimensions=99,
            provider="bedrock", model="x", llm_identifier="id-1",
        )


# ── registry ─────────────────────────────────────────────────────────────

def test_embedding_registry_finds_registered_provider(fake_embedding_provider):
    assert is_embedding_registered(fake_embedding_provider)
    assert fake_embedding_provider in list_embedding_providers()
    assert callable(get_embedding_handler(fake_embedding_provider))


def test_embedding_registry_raises_for_unknown_name():
    with pytest.raises(ProviderNotFoundError):
        get_embedding_handler("totally-unregistered-embedding-model")


def test_titan_embed_v2_is_registered():
    # The real provider added alongside this test suite.
    assert is_embedding_registered("titan-embed-v2")


# ── Router.get_embedding() dispatch ─────────────────────────────────────

@pytest.fixture
def router():
    return Router()


async def test_get_embedding_successful_dispatch_returns_dict(router, fake_embedding_provider):
    result = await router.get_embedding(
        llm_name=fake_embedding_provider,
        text="hello world",
        repo_name="repo",
        llm_identifier="id-1",
        **aws_credentials(),
    )
    assert isinstance(result, dict)
    assert result["dimensions"] == 4
    assert len(result["embedding"]) == 4


async def test_get_embedding_respects_requested_dimensions(router, fake_embedding_provider):
    result = await router.get_embedding(
        llm_name=fake_embedding_provider,
        text="hello world",
        dimensions=16,
        repo_name="repo",
        llm_identifier="id-1",
        **aws_credentials(),
    )
    assert result["dimensions"] == 16


async def test_get_embedding_unregistered_provider_raises_provider_not_found(router):
    with pytest.raises(ProviderNotFoundError):
        await router.get_embedding(
            llm_name="totally-unregistered-embedding-model",
            text="hi",
            repo_name="repo",
            llm_identifier="id-1",
            **aws_credentials(),
        )


async def test_get_embedding_invalid_input_raises_router_validation_error(router, fake_embedding_provider):
    with pytest.raises(RouterValidationError):
        await router.get_embedding(
            llm_name=fake_embedding_provider,
            text="   ",  # empty after strip -> fails EmbeddingRequest validation
            repo_name="repo",
            llm_identifier="id-1",
            **aws_credentials(),
        )


# ── help.py introspection ───────────────────────────────────────────────

def test_list_embedding_models_includes_titan():
    names = [m["llm_name"] for m in list_embedding_models()]
    assert "titan-embed-v2" in names


def test_embedding_model_info_for_titan():
    info = embedding_model_info("titan-embed-v2")
    assert info["provider"] == "bedrock"
    assert info["model_id"] == "amazon.titan-embed-text-v2:0"
    assert "text" in info["required"]
    assert "dimensions" in info["optional"]


def test_embedding_model_info_raises_for_unknown_name():
    with pytest.raises(ProviderNotFoundError):
        embedding_model_info("no-such-embedding-model")
