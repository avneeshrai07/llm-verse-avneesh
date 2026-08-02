from __future__ import annotations

from llm_verse_avneesh.providers.bedrock._regions import resolve_model_id


def test_us_region_gets_us_prefix():
    assert resolve_model_id("amazon.nova-lite-v1:0", "us-east-1") == "us.amazon.nova-lite-v1:0"
    assert resolve_model_id("amazon.nova-lite-v1:0", "us-west-2") == "us.amazon.nova-lite-v1:0"


def test_eu_region_gets_eu_prefix():
    assert resolve_model_id("amazon.nova-lite-v1:0", "eu-west-1") == "eu.amazon.nova-lite-v1:0"


def test_apac_region_gets_apac_prefix():
    assert resolve_model_id("amazon.nova-lite-v1:0", "ap-southeast-2") == "apac.amazon.nova-lite-v1:0"
    assert resolve_model_id("amazon.nova-lite-v1:0", "ap-northeast-1") == "apac.amazon.nova-lite-v1:0"


def test_unmapped_region_falls_back_to_base_model_id():
    # ca-central-1 / sa-east-1 aren't in any known cross-region inference
    # profile grouping for these models -- must not guess a prefix.
    assert resolve_model_id("amazon.nova-lite-v1:0", "ca-central-1") == "amazon.nova-lite-v1:0"
    assert resolve_model_id("amazon.nova-lite-v1:0", "sa-east-1") == "amazon.nova-lite-v1:0"


def test_none_region_falls_back_to_base_model_id():
    assert resolve_model_id("amazon.nova-lite-v1:0", None) == "amazon.nova-lite-v1:0"


def test_works_for_any_base_model_id_not_just_nova():
    assert resolve_model_id(
        "anthropic.claude-haiku-4-5-20251001-v1:0", "us-east-1"
    ) == "us.anthropic.claude-haiku-4-5-20251001-v1:0"
