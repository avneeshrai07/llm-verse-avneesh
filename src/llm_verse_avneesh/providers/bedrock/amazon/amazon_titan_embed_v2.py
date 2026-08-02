from __future__ import annotations
import logging
from langchain_aws import BedrockEmbeddings

from ...._register import register_embedding
from ....models import EmbeddingRequest
from ....types import EmbeddingResponse

logger = logging.getLogger(__name__)
MODEL_ID = "amazon.titan-embed-text-v2:0"
DISPLAY_NAME = "Amazon Titan Embed Text V2"
DEFAULT_DIMENSIONS = 1024


@register_embedding("titan-embed-v2")
async def amazon_titan_embed_v2_function(request: EmbeddingRequest) -> dict:
    """Amazon Titan Embed Text V2 via Bedrock. Supports 256/512/1024 output
    dimensions (default 1024) via request.dimensions."""

    dimensions = request.dimensions or DEFAULT_DIMENSIONS

    client = BedrockEmbeddings(
        model_id=MODEL_ID,
        region_name=request.region_name,
        aws_access_key_id=request.aws_access_key_id,
        aws_secret_access_key=request.aws_secret_access_key.get_secret_value(),
        model_kwargs={"dimensions": dimensions},
    )

    vector = await client.aembed_query(request.text)

    logger.info(
        "titan-embed-v2 | identifier=%s | dimensions=%d",
        request.llm_identifier, len(vector),
    )

    result = EmbeddingResponse(
        embedding=vector,
        dimensions=len(vector),
        provider="bedrock",
        model=MODEL_ID,
        llm_identifier=request.llm_identifier,
    )
    return result.model_dump()
