"""
Regression test for the __all__ missing-comma bug: two adjacent string
literals in a list concatenate into one entry, silently dropping a name
from __all__ (e.g. "LLMResponse" "LLMRouterError" -> one bad entry
instead of two).
"""
from __future__ import annotations

import llm_verse_avneesh as pkg


def test_all_has_no_duplicate_or_merged_entries():
    assert len(pkg.__all__) == len(set(pkg.__all__)), "duplicate entries in __all__"
    for name in pkg.__all__:
        assert name.isidentifier(), f"{name!r} is not a valid identifier — likely a concatenation bug"


def test_all_entries_are_actually_importable():
    for name in pkg.__all__:
        assert hasattr(pkg, name), f"{name!r} listed in __all__ but not defined on the package"


def test_expected_public_names_present():
    expected = {
        "Router",
        "LLMResponse",
        "LLMRouterError",
        "RouterValidationError",
        "ProviderNotFoundError",
        "LLMCallError",
    }
    assert expected.issubset(set(pkg.__all__))
