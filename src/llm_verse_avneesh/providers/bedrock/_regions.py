from __future__ import annotations
"""
Cross-region inference profile prefix resolution, shared across the Bedrock
chat handlers that support it.

Bedrock's cross-region inference profiles route a request to whichever
specific region within a geography currently has capacity, but the profile
id itself is geography-scoped (e.g. "us.amazon.nova-lite-v1:0" only works
when called *from* a US region) - a caller in ap-southeast-2 passing a
"us." profile id gets a hard ValidationException, not automatic routing to
a working region. Each handler used to hardcode the "us." prefix
unconditionally, which broke every non-US caller regardless of what
region_name they passed.
"""

# AWS's own cross-region inference profile geography groupings. Update this
# table if AWS adds a grouping for a region not yet listed here - regions
# not in any known grouping fall back to the base on-demand model id.
_GEOGRAPHY_PREFIXES: dict[str, str] = {
    "us-east-1": "us", "us-east-2": "us", "us-west-1": "us", "us-west-2": "us",
    "eu-central-1": "eu", "eu-west-1": "eu", "eu-west-2": "eu",
    "eu-west-3": "eu", "eu-north-1": "eu",
    "ap-south-1": "apac", "ap-northeast-1": "apac", "ap-northeast-2": "apac",
    "ap-northeast-3": "apac", "ap-southeast-1": "apac", "ap-southeast-2": "apac",
}


def resolve_model_id(base_model_id: str, region_name: str | None) -> str:
    """
    Return the cross-region inference profile id appropriate for
    `region_name`, or `base_model_id` unmodified if the region isn't in a
    known geography grouping (e.g. ca-central-1, sa-east-1) - falls back to
    the plain on-demand model id rather than guessing a profile prefix that
    might not exist for that geography.
    """
    prefix = _GEOGRAPHY_PREFIXES.get(region_name) if region_name else None
    return f"{prefix}.{base_model_id}" if prefix else base_model_id
