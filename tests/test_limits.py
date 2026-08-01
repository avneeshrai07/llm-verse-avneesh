from __future__ import annotations

from llm_verse_avneesh.limits import get_limits, MODEL_LIMITS, _DEFAULT_LIMITS


def test_known_model_has_specific_limits():
    limits = get_limits("gemini-3.1-flash-lite")
    assert limits == MODEL_LIMITS["gemini-3.1-flash-lite"]
    assert limits.temperature_max == 2.0


def test_unknown_model_falls_back_to_default():
    limits = get_limits("some-model-nobody-registered")
    assert limits == _DEFAULT_LIMITS


def test_all_registered_limits_are_positive():
    for name, limits in MODEL_LIMITS.items():
        assert limits.temperature_max > 0, name
        assert limits.max_output_tokens > 0, name
