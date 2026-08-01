# src/llm_verse_avneesh/router.py

from __future__ import annotations

import logging
from typing import TypeVar, Type, Optional, List, Any

from pydantic import BaseModel, SecretStr

from .models import LLMRequest, AWSRegion
from .registry import get_handler
from .limits import get_limits
from .exceptions import (
    RouterValidationError,
    ProviderNotFoundError,
    LLMCallError,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class Router:

    def __init__(self, usage_repo=None):
        self._usage_repo = usage_repo

    async def get_response(
        self,
        llm_name: str,
        system_prompt: str,
        user_prompt: str,
        context: str | None,
        temperature: float,
        pydantic_model: Type[T] | None,
        max_tokens: int,
        repo_name: str,
        llm_identifier: str,
        # ── AWS / Bedrock (optional for Gemini) ───────────────
        region_name: Optional[AWSRegion] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str | SecretStr] = None,
        # ── Google / Gemini (optional for AWS) ────────────────
        google_api_key: Optional[str | SecretStr] = None,
        # ── Agentic ───────────────────────────────────────────
        tools: Optional[List[Any]] = None,
        max_iterations: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Call the model registered as `llm_name` and return its response.

        See llm_verse_avneesh.model_info(llm_name) for which arguments a given
        model actually requires/accepts - e.g. Gemini models need
        google_api_key instead of the AWS credential arguments, and only
        some Bedrock models accept `tools`/`max_iterations`.
        """

        # ── 1. Validate inputs ────────────────────────────────
        try:
            request = LLMRequest(
                llm_name=llm_name,
                region_name=region_name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                context=context,
                temperature=temperature,
                pydantic_model=pydantic_model,
                max_tokens=max_tokens,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                google_api_key=google_api_key,
                repo_name=repo_name,
                llm_identifier=llm_identifier,
                tools=tools,
                max_iterations=max_iterations,
            )
        except Exception as exc:
            raise RouterValidationError(f"Invalid input: {exc}") from exc

        # ── 2. Enforce the requested model's actual limits ────
        # A single global temperature/max_tokens range doesn't reflect
        # what any specific model accepts, so this is checked per
        # llm_name rather than as a blanket rule in LLMRequest.
        limits = get_limits(request.llm_name)
        if request.temperature > limits.temperature_max:
            raise RouterValidationError(
                f"temperature={request.temperature} exceeds the limit of "
                f"{limits.temperature_max} for llm_name={request.llm_name!r}"
            )
        if request.max_tokens > limits.max_output_tokens:
            raise RouterValidationError(
                f"max_tokens={request.max_tokens} exceeds the limit of "
                f"{limits.max_output_tokens} for llm_name={request.llm_name!r}"
            )

        # ── 3. Find the relevant function from registry ───────
        handler = get_handler(request.llm_name)

        # ── 4. Transfer user input to that function ───────────
        try:
            response: dict[str, Any] = await handler(request)
        except ProviderNotFoundError:
            raise
        except Exception as exc:
            logger.error(
                "LLM call failed | llm=%s | identifier=%s | error=%s",
                request.llm_name, request.llm_identifier, exc,
            )
            raise LLMCallError(
                f"'{request.llm_name}' failed for '{request.llm_identifier}': {exc}"
            ) from exc

        # ── 5. Return the LLM output to the user ────────────
        return response
