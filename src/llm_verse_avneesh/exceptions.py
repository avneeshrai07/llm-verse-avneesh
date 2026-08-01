# src/llm_verse_avneesh/exceptions.py


# ─────────────────────────────────────────────────────────────
# BASE
# ─────────────────────────────────────────────────────────────

class LLMRouterError(Exception):
    """
    Base exception for all llm_verse_avneesh errors.

    Catch this to handle every error the package can raise:

        try:
            response = await router.get_response(...)
        except LLMRouterError as e:
            print(f"Something went wrong: {e}")
    """
    pass


# ─────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────

class RouterValidationError(LLMRouterError):
    """
    Raised when the input to Router.get_response() fails validation.

    Causes
    ------
    - Any required field is missing or empty
    - temperature or max_tokens is outside the sane bounds accepted by pydantic
    - temperature or max_tokens exceeds the specific limit of the requested
      llm_name (see llm_verse_avneesh.limits.MODEL_LIMITS)
    - pydantic_model is not a valid Pydantic BaseModel class
    - Credentials required for the requested provider are missing

    Example
    -------
        try:
            await router.get_response(temperature=5.0, ...)
        except RouterValidationError as e:
            print(f"Bad input: {e}")
    """
    pass


# ─────────────────────────────────────────────────────────────
# REGISTRY
# ─────────────────────────────────────────────────────────────

class ProviderNotFoundError(LLMRouterError):
    """
    Raised when llm_name is not registered in the registry.

    Causes
    ------
    - Typo in llm_name e.g. "claude-haiku-4-5" vs "claude-haiku-45"
    - Provider module was never imported so @register() never ran
    - Provider was unregistered (e.g. in tests)

    Example
    -------
        try:
            await router.get_response(llm_name="unknown_llm", ...)
        except ProviderNotFoundError as e:
            print(f"No such provider: {e}")
    """
    pass


# ─────────────────────────────────────────────────────────────
# LLM CALL
# ─────────────────────────────────────────────────────────────

class LLMCallError(LLMRouterError):
    """
    Raised when the LLM provider returns an error or times out.

    Causes
    ------
    - API key invalid or expired
    - Rate limit hit
    - Network timeout
    - Provider returned an unexpected response format
    - Context window exceeded
    - A provider-specific constraint was violated (e.g. Nova grounding
      only being available in US AWS regions)

    Example
    -------
        try:
            await router.get_response(llm_name="gemini-3.1-flash-lite", ...)
        except LLMCallError as e:
            print(f"LLM failed: {e}")
    """
    pass
