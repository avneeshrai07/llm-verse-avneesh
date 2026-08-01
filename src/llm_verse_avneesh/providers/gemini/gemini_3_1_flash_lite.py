from __future__ import annotations
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from ..._register import register
from ...models import LLMRequest
from ...types import LLMResponse

logger = logging.getLogger(__name__)
MODEL_ID = "gemini-3.1-flash-lite"
DISPLAY_NAME = "Gemini 3.1 Flash Lite"
SUPPORTS_STRUCTURED_OUTPUT = True
SUPPORTS_TOOLS = False
NOTES = (
    "No web-search/grounding capability is implemented for Gemini in this "
    "library yet - only plain text and structured output are supported."
)


@register("gemini-3.1-flash-lite")
async def gemini_3_1_flash_lite_function(request: LLMRequest) -> dict:
    """Google Gemini 3.1 Flash Lite. Plain text or structured (pydantic_model) output."""

    client = ChatGoogleGenerativeAI(
        model=MODEL_ID,
        google_api_key=request.google_api_key.get_secret_value(),
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )

    user_message = (
        f"{request.user_prompt}\n\nContext:\n{request.context}"
        if request.context else request.user_prompt
    )
    messages = [
        SystemMessage(content=request.system_prompt),
        HumanMessage(content=user_message),
    ]

    # ── Structured output (pydantic schema provided) ──────────
    if request.pydantic_model is not None:
        structured_client = client.with_structured_output(
            request.pydantic_model,
            method="json_mode",     # more reliable than default on Gemini
            include_raw=True,
        )
        response = await structured_client.ainvoke(messages)
        raw    = response["raw"]
        parsed = response["parsed"]
        usage  = raw.usage_metadata

        logger.info(
            "gemini-3.1-flash-lite [structured] | identifier=%s | input=%d | output=%d",
            request.llm_identifier, usage["input_tokens"], usage["output_tokens"],
        )

        if parsed is None:
            raw_text = raw.content if isinstance(raw.content, str) else str(raw.content)
            raise ValueError(
                f"Gemini structured output returned None for schema "
                f"{request.pydantic_model.__name__}.\n"
                f"Raw response: {raw_text[:500]}"
            )

        result = LLMResponse(
            response=parsed.model_dump(),
            provider="google",
            model=MODEL_ID,
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            llm_identifier=request.llm_identifier,
        )
        return result.model_dump()

    # ── Plain text (no schema) ────────────────────────────────
    response = await client.ainvoke(messages)
    usage    = response.usage_metadata
    raw_text = response.content

    logger.info(
        "gemini-3.1-flash-lite [text] | identifier=%s | input=%d | output=%d",
        request.llm_identifier, usage["input_tokens"], usage["output_tokens"],
    )
    result = LLMResponse(
        response=raw_text,
        provider="google",
        model=MODEL_ID,
        input_tokens=usage["input_tokens"],
        output_tokens=usage["output_tokens"],
        llm_identifier=request.llm_identifier,
    )
    return result.model_dump()
