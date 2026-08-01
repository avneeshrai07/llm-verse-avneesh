# src/llm_verse_avneesh/types.py

from __future__ import annotations

from typing import Union
from pydantic import BaseModel, field_validator


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
