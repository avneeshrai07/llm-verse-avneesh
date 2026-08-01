from __future__ import annotations
import logging
import boto3
from botocore.config import Config

from ...._register import register
from ....models import LLMRequest
from ....types import LLMResponse


logger = logging.getLogger(__name__)
MODEL_ID = "us.amazon.nova-2-lite-v1:0"
DISPLAY_NAME = "Amazon Nova 2 Lite (Grounding)"
SUPPORTS_STRUCTURED_OUTPUT = False
SUPPORTS_TOOLS = False
NOTES = (
    "region_name must be a US AWS region (us-*). Web-search grounding runs "
    "automatically via Nova's built-in systemTool and is not configurable "
    "per-request - there is no `tools` argument for this model."
)

TOOL_CONFIG = {
    "tools": [
        {
            "systemTool": {
                "name": "nova_grounding"
            }
        }
    ]
}


@register("nova-2-lite-grounding")
async def amazon_nova2_lite_grounding_function(request: LLMRequest) -> dict:
    """Amazon Nova 2 Lite via Bedrock with built-in web-search grounding. US AWS regions only; returns cited sources alongside the response."""

    # Grounding only works in US AWS regions. Rather than silently forcing
    # us-east-1 regardless of what the caller asked for, honor
    # request.region_name and fail loudly if it's outside the supported set.
    if not request.region_name.startswith("us-"):
        raise ValueError(
            f"nova-2-lite-grounding only supports US AWS regions "
            f"(got region_name={request.region_name!r})"
        )

    # Must use boto3 directly - ChatBedrockConverse does NOT support systemTool/toolConfig
    bedrock = boto3.client(
        "bedrock-runtime",
        region_name=request.region_name,
        aws_access_key_id=request.aws_access_key_id,
        aws_secret_access_key=request.aws_secret_access_key.get_secret_value(),
        config=Config(read_timeout=3600),      # Extended timeout required for web searches
    )

    user_message = (
        f"{request.user_prompt}\n\nContext:\n{request.context}"
        if request.context else request.user_prompt
    )

    messages = [
        {
            "role": "user",
            "content": [{"text": user_message}]
        }
    ]

    # system prompt goes in the `system` param, not as a message
    system = [{"text": request.system_prompt}] if request.system_prompt else []

    response = bedrock.converse(
        modelId=MODEL_ID,
        messages=messages,
        system=system,
        toolConfig=TOOL_CONFIG,
        inferenceConfig={
            "temperature": request.temperature,
            "maxTokens": request.max_tokens,
        },
    )

    usage = response["usage"]

    content_list = response["output"]["message"]["content"]

    text_parts = [
        block["text"] for block in content_list
        if isinstance(block, dict) and "text" in block
    ]

    citations = [
        citation["location"]["web"]["url"]
        for block in content_list
        if isinstance(block, dict) and "citationsContent" in block
        for citation in block["citationsContent"]["citations"]
    ]

    raw_text = {
        "text": " ".join(text_parts),
        "citations": citations,
    }

    logger.info(
        "nova-2-lite-grounding [text] | identifier=%s | input=%d | output=%d",
        request.llm_identifier,
        usage["inputTokens"],
        usage["outputTokens"],
    )

    result = LLMResponse(
        response=raw_text,
        provider="bedrock",
        model=MODEL_ID,
        input_tokens=usage["inputTokens"],
        output_tokens=usage["outputTokens"],
        llm_identifier=request.llm_identifier,
    )
    return result.model_dump()
