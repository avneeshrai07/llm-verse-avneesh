# src/llm_verse_avneesh/help.py

from __future__ import annotations

import inspect
import sys
from typing import Any

from .registry import (
    get_handler,
    list_providers,
    get_embedding_handler,
    list_embedding_providers,
)
from .router import Router

# Fields every model needs regardless of provider - part of LLMRequest's
# universal contract (see models.py), not a per-model detail.
_BASE_REQUIRED = {
    "system_prompt": "The system/instruction prompt for the model.",
    "user_prompt": "The user's prompt/question.",
    "temperature": "Sampling temperature (capped per-model - see llm_verse_avneesh.limits).",
    "max_tokens": "Maximum output tokens (capped per-model - see llm_verse_avneesh.limits).",
    "repo_name": "Caller-supplied repo/project identifier, echoed back for tracing.",
    "llm_identifier": "Caller-supplied identifier for this call, echoed back in the response.",
}

_BASE_OPTIONAL = {
    "context": "Extra context appended after the user prompt.",
}

_AWS_REQUIRED = {
    "region_name": "AWS region to call Bedrock in.",
    "aws_access_key_id": "AWS access key id.",
    "aws_secret_access_key": "AWS secret access key.",
}

_GOOGLE_REQUIRED = {
    "google_api_key": "Google API key for the Gemini API.",
}

_GROQ_REQUIRED = {
    "groq_api_key": "Groq API key for the Groq API.",
}

_STRUCTURED_OPTIONAL = {
    "pydantic_model": "A BaseModel subclass - if given, the response is parsed into it instead of returned as plain text.",
}

_TOOLS_OPTIONAL = {
    "tools": "A list of LangChain-compatible tools. If given, the call runs an agentic tool-calling loop instead of a single response.",
    "max_iterations": "Max tool-calling loop iterations (default 10, max 50). Only meaningful when `tools` is given.",
}

_VISION_OPTIONAL = {
    "images": "A list of image URLs or base64 data URIs for vision input. Ignored by non-vision models.",
}


def _provider_category(handler: Any) -> str:
    """Derive the provider folder (e.g. 'bedrock', 'gemini') from the handler's module path."""
    parts = handler.__module__.split(".")
    try:
        return parts[parts.index("providers") + 1]
    except (ValueError, IndexError):
        return "unknown"


def _module_of(handler: Any):
    return sys.modules[handler.__module__]


def list_models() -> list[dict[str, str]]:
    """List every registered model's llm_name, display name, and provider."""
    result = []
    for name in list_providers():
        handler = get_handler(name)
        module = _module_of(handler)
        result.append({
            "llm_name": name,
            "display_name": getattr(module, "DISPLAY_NAME", name),
            "provider": _provider_category(handler),
        })
    return result


def model_info(llm_name: str) -> dict[str, Any]:
    """
    Return what it takes to call a specific model: display name, provider,
    model id, required fields, and the optional fields it actually supports
    (structured output / agentic tools vary by model - see `notes`).
    """
    handler = get_handler(llm_name)  # raises ProviderNotFoundError if unknown
    module = _module_of(handler)
    provider = _provider_category(handler)

    required = dict(_BASE_REQUIRED)
    if provider == "gemini":
        required.update(_GOOGLE_REQUIRED)
    elif provider == "groq":
        required.update(_GROQ_REQUIRED)
    else:
        required.update(_AWS_REQUIRED)

    optional = dict(_BASE_OPTIONAL)
    if getattr(module, "SUPPORTS_STRUCTURED_OUTPUT", False):
        optional.update(_STRUCTURED_OPTIONAL)
    if getattr(module, "SUPPORTS_TOOLS", False):
        optional.update(_TOOLS_OPTIONAL)
    if getattr(module, "SUPPORTS_VISION", False):
        optional.update(_VISION_OPTIONAL)

    return {
        "llm_name": llm_name,
        "display_name": getattr(module, "DISPLAY_NAME", llm_name),
        "provider": provider,
        "model_id": getattr(module, "MODEL_ID", None),
        "description": inspect.getdoc(handler),
        "required": required,
        "optional": optional,
        "notes": getattr(module, "NOTES", None),
    }


_EMBEDDING_AWS_REQUIRED = {
    "region_name": "AWS region to call Bedrock in.",
    "aws_access_key_id": "AWS access key id.",
    "aws_secret_access_key": "AWS secret access key.",
}

_EMBEDDING_BASE_REQUIRED = {
    "text": "The text to embed.",
    "repo_name": "Caller-supplied repo/project identifier, echoed back for tracing.",
    "llm_identifier": "Caller-supplied identifier for this call, echoed back in the response.",
}

_EMBEDDING_BASE_OPTIONAL = {
    "dimensions": "Output vector size, if the model supports more than one (e.g. Titan Embed V2: 256/512/1024). Defaults to the provider's own default.",
}


def list_embedding_models() -> list[dict[str, str]]:
    """List every registered embedding model's llm_name, display name, and provider."""
    result = []
    for name in list_embedding_providers():
        handler = get_embedding_handler(name)
        module = _module_of(handler)
        result.append({
            "llm_name": name,
            "display_name": getattr(module, "DISPLAY_NAME", name),
            "provider": _provider_category(handler),
        })
    return result


def embedding_model_info(llm_name: str) -> dict[str, Any]:
    """
    Return what it takes to call a specific embedding model: display name,
    provider, model id, and required/optional fields.
    """
    handler = get_embedding_handler(llm_name)  # raises ProviderNotFoundError if unknown
    module = _module_of(handler)
    provider = _provider_category(handler)

    required = dict(_EMBEDDING_BASE_REQUIRED)
    required.update(_EMBEDDING_AWS_REQUIRED)  # only AWS/Bedrock is registered so far

    return {
        "llm_name": llm_name,
        "display_name": getattr(module, "DISPLAY_NAME", llm_name),
        "provider": provider,
        "model_id": getattr(module, "MODEL_ID", None),
        "description": inspect.getdoc(handler),
        "required": required,
        "optional": dict(_EMBEDDING_BASE_OPTIONAL),
        "notes": getattr(module, "NOTES", None),
    }


def help() -> None:
    """Print every callable this library exposes and what it's for."""
    print("llm_verse_avneesh - available callables")
    print("=" * 60)
    print()

    print("list_models()")
    print(f"    {inspect.getdoc(list_models)}")
    print()

    print("model_info(llm_name)")
    print(f"    {inspect.getdoc(model_info)}")
    print()

    print("Router().get_response(...)")
    doc = inspect.getdoc(Router.get_response) or "No description available."
    for line in doc.splitlines():
        print(f"    {line}")
    print()

    print("list_embedding_models()")
    print(f"    {inspect.getdoc(list_embedding_models)}")
    print()

    print("embedding_model_info(llm_name)")
    print(f"    {inspect.getdoc(embedding_model_info)}")
    print()

    print("Router().get_embedding(...)")
    doc = inspect.getdoc(Router.get_embedding) or "No description available."
    for line in doc.splitlines():
        print(f"    {line}")
    print()

    print(f"See list_models() for the {len(list_providers())} registered llm_name values, "
          f"and model_info(llm_name) for what each one needs.")
    print(f"See list_embedding_models() for the {len(list_embedding_providers())} registered "
          f"embedding llm_name values, and embedding_model_info(llm_name) for what each needs.")
