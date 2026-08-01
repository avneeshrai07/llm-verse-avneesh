from __future__ import annotations
import logging
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage, SystemMessage

from ..._register import register
from ...models import LLMRequest
from ...types import LLMResponse

logger = logging.getLogger(__name__)
MODEL_ID = "us.amazon.nova-2-lite-v1:0"


@register("nova-2-lite")
async def amazon_nova2_lite_function(request: LLMRequest) -> dict:

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
        logger.info("nova-2-lite [structured] | identifier=%s | input=%d | output=%d",
                    request.llm_identifier, usage["input_tokens"], usage["output_tokens"])
        result = LLMResponse(
            response=parsed.model_dump(),
            provider="bedrock", model=MODEL_ID,
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            llm_identifier=request.llm_identifier,
        )
        return result.model_dump()

    response = await client.ainvoke(messages)
    usage = response.usage_metadata
    raw_text = (
        response.content if isinstance(response.content, str)
        else "".join(
            block.get("text", "") for block in response.content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    )
    logger.info("nova-2-lite [text] | identifier=%s | input=%d | output=%d",
                request.llm_identifier, usage["input_tokens"], usage["output_tokens"])
    result = LLMResponse(
        response=raw_text,
        provider="bedrock", model=MODEL_ID,
        input_tokens=usage["input_tokens"],
        output_tokens=usage["output_tokens"],
        llm_identifier=request.llm_identifier,
    )
    return result.model_dump()
