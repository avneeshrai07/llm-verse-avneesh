# src/llm_verse_avneesh/limits.py
"""
Per-model validation limits.

The old blanket "temperature in [0, 1], max_tokens <= 128000 for every
model" check didn't reflect what any single provider actually accepts —
it silently allowed model+parameter combinations the API would later
reject, and rejected valid values for models that permit more headroom
(e.g. Gemini accepts temperature up to 2.0).

Limits here are keyed by the exact registry name used to dispatch the
request (see registry.py). Treat these as reasonable defaults — confirm
against the provider's current docs before relying on them for a new
model, and update this table when a provider changes its limits.
"""

from __future__ import annotations

from typing import NamedTuple


class ModelLimits(NamedTuple):
    temperature_max: float
    max_output_tokens: int


_DEFAULT_LIMITS = ModelLimits(temperature_max=1.0, max_output_tokens=4096)

MODEL_LIMITS: dict[str, ModelLimits] = {
    "claude-haiku-4-5": ModelLimits(temperature_max=1.0, max_output_tokens=64000),
    "claude-haiku-4-5-agentic": ModelLimits(temperature_max=1.0, max_output_tokens=64000),
    "nova-lite": ModelLimits(temperature_max=1.0, max_output_tokens=5000),
    "nova-2-lite": ModelLimits(temperature_max=1.0, max_output_tokens=5000),
    "nova-2-lite-grounding": ModelLimits(temperature_max=1.0, max_output_tokens=5000),
    "nova-pro": ModelLimits(temperature_max=1.0, max_output_tokens=5000),
    "gemini-3.1-flash-lite": ModelLimits(temperature_max=2.0, max_output_tokens=8192),
}


def get_limits(llm_name: str) -> ModelLimits:
    """Return the known limits for llm_name, or a conservative default."""
    return MODEL_LIMITS.get(llm_name, _DEFAULT_LIMITS)
