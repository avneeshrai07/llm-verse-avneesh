# src/llm_verse_avneesh/models.py

from __future__ import annotations

from typing import Type, Any, Literal, Optional, List
from pydantic import BaseModel, SecretStr, model_validator, field_validator


# ─────────────────────────────────────────────────────────────
# VALID AWS REGIONS
# ─────────────────────────────────────────────────────────────

AWSRegion = Literal[
    # United States
    "us-east-1",       # N. Virginia
    "us-east-2",       # Ohio
    "us-west-1",       # N. California
    "us-west-2",       # Oregon
    # Asia Pacific
    "ap-south-1",      # Mumbai
    "ap-northeast-1",  # Tokyo
    "ap-northeast-2",  # Seoul
    "ap-northeast-3",  # Osaka
    "ap-southeast-1",  # Singapore
    "ap-southeast-2",  # Sydney
    # Canada
    "ca-central-1",    # Central
    # Europe
    "eu-central-1",    # Frankfurt
    "eu-west-1",       # Ireland
    "eu-west-2",       # London
    "eu-west-3",       # Paris
    "eu-north-1",      # Stockholm
    # South America
    "sa-east-1",       # São Paulo
]


# ─────────────────────────────────────────────────────────────
# GROQ REGISTRY NAMES
# ─────────────────────────────────────────────────────────────
# Groq model names don't share a common substring the way "gemini" does,
# so credential validation below keys off this explicit set instead of a
# name-pattern check.

GROQ_MODEL_NAMES = frozenset({
    "gpt-oss-120b",
    "gpt-oss-20b",
    "qwen-3.6-27b",
})


# ─────────────────────────────────────────────────────────────
# REQUEST MODEL
# ─────────────────────────────────────────────────────────────

class LLMRequest(BaseModel):
    llm_name: str
    region_name: Optional[AWSRegion] = None    # optional: not needed for Gemini/Groq
    system_prompt: str
    user_prompt: str
    context: str | None = None
    temperature: float
    pydantic_model: Type[BaseModel] | None = None
    max_tokens: int
    # SecretStr: prevents credentials leaking into repr(), logs, or
    # model_dump()/model_dump_json() output by accident. Call
    # .get_secret_value() at the point of use (inside a provider handler).
    aws_access_key_id: Optional[str] = None    # optional: not needed for Gemini/Groq
    aws_secret_access_key: Optional[SecretStr] = None  # optional: not needed for Gemini/Groq
    google_api_key: Optional[SecretStr] = None       # optional: not needed for AWS/Groq
    groq_api_key: Optional[SecretStr] = None         # optional: not needed for AWS/Gemini
    repo_name: str
    llm_identifier: str
    tools: Optional[List[Any]] = None
    max_iterations: Optional[int] = None
    # Image URLs or base64 data URIs (e.g. "data:image/jpeg;base64,...") for
    # vision-capable models. Ignored by models that don't support vision.
    images: Optional[List[str]] = None

    model_config = {"arbitrary_types_allowed": True}

    # Only validate fields that are always required and always strings
    @field_validator("llm_name", "system_prompt", "user_prompt",
                     "repo_name", "llm_identifier")
    @classmethod
    def validate_non_empty_strings(cls, v: str, info: Any) -> str:
        if not v or not v.strip():
            raise ValueError(f"{info.field_name!r} must be a non-empty string")
        return v.strip()

    # Validate optional string fields only when provided
    @field_validator("aws_access_key_id")
    @classmethod
    def validate_optional_strings(cls, v: Optional[str], info: Any) -> Optional[str]:
        if v is not None and not v.strip():
            raise ValueError(f"{info.field_name!r} must not be an empty string if provided")
        return v.strip() if v else None

    # Same emptiness check, but for the SecretStr credential fields.
    @field_validator("aws_secret_access_key", "google_api_key", "groq_api_key")
    @classmethod
    def validate_optional_secrets(cls, v: Optional[SecretStr], info: Any) -> Optional[SecretStr]:
        if v is not None and not v.get_secret_value().strip():
            raise ValueError(f"{info.field_name!r} must not be an empty string if provided")
        return v

    @field_validator("images")
    @classmethod
    def validate_images(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is None:
            return v
        if not v:
            raise ValueError("images must not be an empty list if provided")
        if any(not img or not img.strip() for img in v):
            raise ValueError("images must not contain empty strings")
        return v

    # Sanity bounds only — these are NOT the real per-model limits.
    # Router.get_response() enforces the actual per-llm_name bounds from
    # llm_verse_avneesh.limits once llm_name is known, since a single
    # global range doesn't reflect what any given model actually accepts.
    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if v < 0.0:
            raise ValueError(f"temperature must be non-negative, got {v}")
        return v

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"max_tokens must be a positive integer, got {v}")
        return v

    @field_validator("max_iterations")
    @classmethod
    def validate_max_iterations(cls, v: Optional[int]) -> Optional[int]:
        if v is None:
            return v
        if v <= 0:
            raise ValueError(f"max_iterations must be a positive integer, got {v}")
        if v > 50:
            raise ValueError(f"max_iterations exceeds upper limit of 50, got {v}")
        return v

    @field_validator("context")
    @classmethod
    def validate_context(cls, v: str | None) -> str | None:
        if v is not None and not v.strip():
            return None
        return v

    @model_validator(mode="after")
    def validate_pydantic_model_is_basemodel(self) -> "LLMRequest":
        if self.pydantic_model is not None:
            if not (isinstance(self.pydantic_model, type) and
                    issubclass(self.pydantic_model, BaseModel)):
                raise ValueError(
                    f"pydantic_model must be a BaseModel class, "
                    f"got {type(self.pydantic_model)}"
                )
        return self

    @model_validator(mode="after")
    def validate_provider_credentials(self) -> "LLMRequest":
        """Ensure the right credentials are present for the chosen provider."""
        is_gemini = "gemini" in self.llm_name.lower()
        is_groq = self.llm_name in GROQ_MODEL_NAMES

        if is_gemini:
            if not self.google_api_key:
                raise ValueError(
                    "google_api_key is required for Gemini models"
                )
        elif is_groq:
            if not self.groq_api_key:
                raise ValueError(
                    "groq_api_key is required for Groq models"
                )
        else:
            if not self.aws_access_key_id or not self.aws_secret_access_key:
                raise ValueError(
                    "aws_access_key_id and aws_secret_access_key are required for AWS/Bedrock models"
                )
            if not self.region_name:
                raise ValueError(
                    "region_name is required for AWS/Bedrock models"
                )
        return self
