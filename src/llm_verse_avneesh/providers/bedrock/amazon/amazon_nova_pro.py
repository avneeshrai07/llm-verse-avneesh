from __future__ import annotations
import logging
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from ...._register import register
from ....models import LLMRequest
from ....types import LLMResponse

logger = logging.getLogger(__name__)
MODEL_ID = "us.amazon.nova-pro-v1:0"
DISPLAY_NAME = "Amazon Nova Pro"
SUPPORTS_STRUCTURED_OUTPUT = True
SUPPORTS_TOOLS = True


@register("nova-pro")
async def amazon_nova_pro_function(request: LLMRequest) -> dict:
    """Amazon Nova Pro via Bedrock. Plain text or structured (pydantic_model) output; pass tools=[...] to run an agentic tool-calling loop."""

    client = ChatBedrockConverse(
        model_id=MODEL_ID,
        region_name=request.region_name,
        aws_access_key_id=request.aws_access_key_id,
        aws_secret_access_key=request.aws_secret_access_key.get_secret_value(),
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

    if request.pydantic_model is not None:
        structured_client = client.with_structured_output(
            request.pydantic_model, include_raw=True
        )
        response = await structured_client.ainvoke(messages)
        raw, parsed, usage = response["raw"], response["parsed"], response["raw"].usage_metadata
        logger.info("nova-pro [structured] | identifier=%s | input=%d | output=%d",
                    request.llm_identifier, usage["input_tokens"], usage["output_tokens"])
        result = LLMResponse(
            response=parsed.model_dump(),
            provider="bedrock", model=MODEL_ID,
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            llm_identifier=request.llm_identifier,
        )
        return result.model_dump()

    # Passing tools=[...] is what turns this call agentic — no separate
    # registration/model id for a "nova-pro-agentic" variant.
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
            "nova-pro | identifier=%s | iteration=%d/%d | %d tool call(s): %s",
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
                        logger.error("nova-pro | tool error | %s | %s", tool_name, exc)
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

    # Nova sometimes echoes its internal tool-result wrapper verbatim in the
    # final answer instead of just the summarized text.
    stripped = raw_text.strip()
    if stripped.startswith("<toolResult>") and stripped.endswith("</toolResult>"):
        raw_text = stripped[len("<toolResult>"):-len("</toolResult>")].strip()

    logger.info("nova-pro [text] | identifier=%s | input=%d | output=%d",
                request.llm_identifier, total_input_tokens, total_output_tokens)
    result = LLMResponse(
        response=raw_text,
        provider="bedrock", model=MODEL_ID,
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        llm_identifier=request.llm_identifier,
    )
    return result.model_dump()
