from .router import Router
from .types import LLMResponse
from .exceptions import (
    LLMRouterError,
    RouterValidationError,
    ProviderNotFoundError,
    LLMCallError,
)
from .help import help, list_models, model_info

# CRITICAL: triggers @register() decorators
from .providers.bedrock.claude import claude_haiku_4_5      # noqa: F401
from .providers.bedrock.amazon import amazon_nova_lite              # noqa: F401
from .providers.bedrock.amazon import amazon_nova2_lite             # noqa: F401
from .providers.bedrock.amazon import amazon_nova_pro                # noqa: F401
from .providers.bedrock.amazon import amazon_nova2_lite_grounding    # noqa: F401
from .providers.gemini import gemini_3_1_flash_lite          # noqa: F401

__all__ = [
    "Router",
    "LLMResponse",
    "LLMRouterError",
    "RouterValidationError",
    "ProviderNotFoundError",
    "LLMCallError",
    "help",
    "list_models",
    "model_info",
]
