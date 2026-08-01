from __future__ import annotations

import pytest
from pydantic import BaseModel, SecretStr, ValidationError

from llm_verse_avneesh.models import LLMRequest
from tests.conftest import aws_credentials


def make_request(**overrides):
    base = dict(
        llm_name="nova-lite",
        system_prompt="be helpful",
        user_prompt="hello",
        context=None,
        temperature=0.5,
        pydantic_model=None,
        max_tokens=100,
        repo_name="repo",
        llm_identifier="id-1",
        **aws_credentials(),
    )
    base.update(overrides)
    return LLMRequest(**base)


def test_valid_request_builds():
    req = make_request()
    assert req.llm_name == "nova-lite"


def test_negative_temperature_rejected():
    with pytest.raises(ValidationError):
        make_request(temperature=-0.1)


def test_zero_max_tokens_rejected():
    with pytest.raises(ValidationError):
        make_request(max_tokens=0)


def test_empty_system_prompt_rejected():
    with pytest.raises(ValidationError):
        make_request(system_prompt="   ")


def test_missing_aws_credentials_rejected():
    with pytest.raises(ValidationError):
        LLMRequest(
            llm_name="nova-lite",
            system_prompt="x",
            user_prompt="y",
            context=None,
            temperature=0.5,
            pydantic_model=None,
            max_tokens=100,
            repo_name="repo",
            llm_identifier="id-1",
        )


def test_missing_google_api_key_rejected_for_gemini():
    with pytest.raises(ValidationError):
        LLMRequest(
            llm_name="gemini-3.1-flash-lite",
            system_prompt="x",
            user_prompt="y",
            context=None,
            temperature=0.5,
            pydantic_model=None,
            max_tokens=100,
            repo_name="repo",
            llm_identifier="id-1",
        )


def test_missing_groq_api_key_rejected_for_groq():
    with pytest.raises(ValidationError):
        LLMRequest(
            llm_name="gpt-oss-120b",
            system_prompt="x",
            user_prompt="y",
            context=None,
            temperature=0.5,
            pydantic_model=None,
            max_tokens=100,
            repo_name="repo",
            llm_identifier="id-1",
        )


def test_groq_api_key_accepted_for_groq_model():
    req = LLMRequest(
        llm_name="gpt-oss-120b",
        system_prompt="x",
        user_prompt="y",
        context=None,
        temperature=0.5,
        pydantic_model=None,
        max_tokens=100,
        repo_name="repo",
        llm_identifier="id-1",
        groq_api_key="fake-groq-key",
    )
    assert req.groq_api_key.get_secret_value() == "fake-groq-key"


def test_empty_images_list_rejected():
    with pytest.raises(ValidationError):
        make_request(images=[])


def test_images_with_blank_entry_rejected():
    with pytest.raises(ValidationError):
        make_request(images=["https://example.com/a.png", "   "])


def test_pydantic_model_must_be_basemodel_subclass():
    with pytest.raises(ValidationError):
        make_request(pydantic_model=int)


def test_whitespace_only_context_becomes_none():
    req = make_request(context="   ")
    assert req.context is None


def test_secret_fields_are_secretstr_and_not_leaked():
    req = make_request()
    assert isinstance(req.aws_secret_access_key, SecretStr)
    # str()/repr() must never expose the raw secret value.
    assert "fake-secret-value-not-real" not in str(req)
    assert "fake-secret-value-not-real" not in repr(req)
    dumped = req.model_dump()
    assert "fake-secret-value-not-real" not in str(dumped)
    assert req.aws_secret_access_key.get_secret_value() == "fake-secret-value-not-real"


def test_structured_pydantic_model_accepted():
    class Schema(BaseModel):
        answer: str

    req = make_request(pydantic_model=Schema)
    assert req.pydantic_model is Schema
