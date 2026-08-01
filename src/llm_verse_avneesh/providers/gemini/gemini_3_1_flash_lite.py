from __future__ import annotations
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from ..._register import register
from ...models import LLMRequest
from ...types import LLMResponse

logger = logging.getLogger(__name__)
MODEL_ID = "gemini-3.1-flash-lite"
DISPLAY_NAME = "Gemini 3.1 Flash Lite"
SUPPORTS_STRUCTURED_OUTPUT = True
SUPPORTS_TOOLS = True
NOTES = (
    "No built-in web-search/grounding tool is wired up for Gemini in this "
    "library (unlike nova-2-lite-grounding) - only user-supplied tools=[...] "
    "are supported."
)


@register("gemini-3.1-flash-lite")
async def gemini_3_1_flash_lite_function(request: LLMRequest) -> dict:
    """Google Gemini 3.1 Flash Lite. Plain text or structured (pydantic_model) output; pass tools=[...] to run an agentic tool-calling loop."""

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

    # ── Plain text, or an agentic tool-calling loop when tools are given ──
    # Passing tools=[...] is what turns this call agentic - no separate
    # registration/model id for a "gemini-3.1-flash-lite-agentic" variant.
    if request.tools:
        client = client.bind_tools(request.tools)

    max_iterations = request.max_iterations if request.max_iterations is not None else 10
    total_input_tokens = 0
    total_output_tokens = 0
    final_response = None

    for iteration in range(max_iterations):
        response = await client.ainvoke(messages)
        usage = response.usage_metadata or {}
        total_input_tokens += usage.get("input_tokens", 0)
        total_output_tokens += usage.get("output_tokens", 0)
        messages.append(response)
        final_response = response

        if not response.tool_calls:
            break

        logger.info(
            "gemini-3.1-flash-lite | identifier=%s | iteration=%d/%d | %d tool call(s): %s",
            request.llm_identifier, iteration + 1, max_iterations,
            len(response.tool_calls), [tc["name"] for tc in response.tool_calls],
        )

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            tool_result = f"Tool '{tool_name}' not found"
            for tool in request.tools or []:
                if getattr(tool, "name", None) == tool_name:
                    try:
                        if hasattr(tool, "ainvoke"):
                            tool_result = await tool.ainvoke(tool_args)
                        else:
                            tool_result = tool.invoke(tool_args)
                    except Exception as exc:
                        logger.error("gemini-3.1-flash-lite | tool error | %s | %s", tool_name, exc)
                        tool_result = f"Error executing tool '{tool_name}': {exc}"
                    break

            messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_id))

    raw_text = ""
    if final_response is not None:
        raw_text = (
            final_response.content if isinstance(final_response.content, str)
            else "".join(
                block.get("text", "") for block in final_response.content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        )

    logger.info(
        "gemini-3.1-flash-lite [text] | identifier=%s | input=%d | output=%d",
        request.llm_identifier, total_input_tokens, total_output_tokens,
    )
    result = LLMResponse(
        response=raw_text,
        provider="google",
        model=MODEL_ID,
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        llm_identifier=request.llm_identifier,
    )
    return result.model_dump()
