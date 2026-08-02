# src/llm_verse_avneesh/types.py

from __future__ import annotations

from typing import Union
from pydantic import BaseModel, field_validator, model_validator


# ─────────────────────────────────────────────────────────────
# LLM RESPONSE
# ─────────────────────────────────────────────────────────────

class LLMResponse(BaseModel):
    """
    Unified response object returned by every provider.

    Every provider's handler MUST build one of these and return
    ``.model_dump()`` — no matter which LLM is called underneath, the
    Router always hands the caller back the same dict shape.

    Fields
    ------
    response        : str for plain text responses, dict for structured
                      (pydantic_model) responses
    provider        : Which backend handled this call (e.g. "bedrock", "google")
    model           : Exact model id used, e.g. "us.amazon.nova-lite-v1:0"
    input_tokens    : Tokens consumed by the prompt
    output_tokens   : Tokens consumed by the response
    llm_identifier  : Echo of the request's llm_identifier — for tracing
    latency_ms      : How long the LLM call took (set by Router, not provider)
    """

    response: Union[str, dict]      # str for text, dict for structured output
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    llm_identifier: str
    latency_ms: int = 0

    @field_validator("response")
    @classmethod
    def response_must_not_be_empty(cls, v: Union[str, dict]) -> Union[str, dict]:
        if not v:
            raise ValueError("LLM returned an empty response")
        return v

    @field_validator("input_tokens", "output_tokens", "latency_ms")
    @classmethod
    def must_be_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Token counts and latency must be non-negative")
        return v

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def is_truncated(self) -> bool:
        return self.output_tokens % 100 == 0 and self.output_tokens > 0

    @property
    def is_structured(self) -> bool:
        """True if this response contains structured (pydantic) output."""
        return isinstance(self.response, dict)

    def __repr__(self) -> str:
        return (
            f"LLMResponse("
            f"provider={self.provider!r}, "
            f"model={self.model!r}, "
            f"tokens={self.total_tokens}, "
            f"latency_ms={self.latency_ms}, "
            f"identifier={self.llm_identifier!r})"
        )


# ─────────────────────────────────────────────────────────────
# EMBEDDING RESPONSE
# ─────────────────────────────────────────────────────────────

class EmbeddingResponse(BaseModel):
    """
    Unified response object returned by every embedding provider.

    Every embedding provider's handler MUST build one of these and return
    ``.model_dump()`` — same convention as LLMResponse, kept as a separate
    type since the shape (a vector, not text/structured output) is
    fundamentally different.

    Fields
    ------
    embedding       : the embedding vector
    dimensions      : len(embedding) — validated to actually match
    provider        : which backend handled this call (e.g. "bedrock")
    model           : exact model id used, e.g. "amazon.titan-embed-text-v2:0"
    input_tokens    : tokens consumed by the input text (0 if the provider
                      doesn't report this)
    llm_identifier  : echo of the request's llm_identifier — for tracing
    latency_ms      : how long the call took (set by Router, not provider)
    """

    embedding: list[float]
    dimensions: int
    provider: str
    model: str
    input_tokens: int = 0
    llm_identifier: str
    latency_ms: int = 0

    @field_validator("embedding")
    @classmethod
    def embedding_must_not_be_empty(cls, v: list[float]) -> list[float]:
        if not v:
            raise ValueError("Embedding provider returned an empty vector")
        return v

    @field_validator("input_tokens", "latency_ms")
    @classmethod
    def must_be_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Token counts and latency must be non-negative")
        return v

    @model_validator(mode="after")
    def dimensions_must_match_embedding_length(self) -> "EmbeddingResponse":
        if self.dimensions != len(self.embedding):
            raise ValueError(
                f"dimensions={self.dimensions} does not match "
                f"len(embedding)={len(self.embedding)}"
            )
        return self

    def __repr__(self) -> str:
        return (
            f"EmbeddingResponse("
            f"provider={self.provider!r}, "
            f"model={self.model!r}, "
            f"dimensions={self.dimensions}, "
            f"latency_ms={self.latency_ms}, "
            f"identifier={self.llm_identifier!r})"
        )
