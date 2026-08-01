from __future__ import annotations

import pytest

import llm_verse_avneesh  # noqa: F401  -- triggers provider registration
from llm_verse_avneesh.exceptions import ProviderNotFoundError
from llm_verse_avneesh.registry import list_providers


def test_list_models_lists_every_registered_provider():
    names = {entry["llm_name"] for entry in llm_verse_avneesh.list_models()}
    assert names == set(list_providers())


def test_list_models_entries_have_provider_and_display_name():
    for entry in llm_verse_avneesh.list_models():
        assert entry["provider"] in {"bedrock", "gemini"}
        assert entry["display_name"]


def test_model_info_raises_for_unknown_llm_name():
    with pytest.raises(ProviderNotFoundError):
        llm_verse_avneesh.model_info("does-not-exist")


def test_model_info_bedrock_requires_aws_not_google():
    info = llm_verse_avneesh.model_info("nova-lite")
    assert "region_name" in info["required"]
    assert "aws_access_key_id" in info["required"]
    assert "aws_secret_access_key" in info["required"]
    assert "google_api_key" not in info["required"]


def test_model_info_gemini_requires_google_not_aws():
    info = llm_verse_avneesh.model_info("gemini-3.1-flash-lite")
    assert "google_api_key" in info["required"]
    assert "region_name" not in info["required"]
    assert "aws_access_key_id" not in info["required"]
    assert "aws_secret_access_key" not in info["required"]


def test_model_info_tool_capable_models_expose_tools_option():
    for name in ("claude-haiku-4-5", "nova-lite", "nova-2-lite", "nova-pro", "gemini-3.1-flash-lite"):
        info = llm_verse_avneesh.model_info(name)
        assert "tools" in info["optional"]
        assert "max_iterations" in info["optional"]


def test_model_info_grounding_has_no_tools_or_structured_option():
    info = llm_verse_avneesh.model_info("nova-2-lite-grounding")
    assert "tools" not in info["optional"]
    assert "pydantic_model" not in info["optional"]
    assert info["notes"] is not None


def test_help_runs_and_mentions_public_callables(capsys):
    llm_verse_avneesh.help()
    output = capsys.readouterr().out

    assert "list_models()" in output
    assert "model_info(llm_name)" in output
    assert "get_response" in output
