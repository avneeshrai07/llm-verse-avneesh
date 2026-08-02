from .router import Router
from .types import LLMResponse, EmbeddingResponse
from .exceptions import (
    LLMRouterError,
    RouterValidationError,
    ProviderNotFoundError,
    LLMCallError,
)
from .help import (
    help,
    list_models,
    model_info,
    list_embedding_models,
    embedding_model_info,
)

# CRITICAL: triggers @register()/@register_embedding() decorators. Each
# provider family's SDK is an optional extra (aws/google/groq in
# pyproject.toml) -- guard each group with try/except ImportError so
# installing only one extra doesn't require the others' SDKs to be present.
# An unavailable provider is simply absent from list_models()/registry
# lookups rather than breaking `import llm_verse_avneesh` entirely.
try:
    from .providers.bedrock.claude import claude_haiku_4_5      # noqa: F401
    from .providers.bedrock.amazon import amazon_nova_lite              # noqa: F401
    from .providers.bedrock.amazon import amazon_nova2_lite             # noqa: F401
    from .providers.bedrock.amazon import amazon_nova_pro                # noqa: F401
    from .providers.bedrock.amazon import amazon_nova2_lite_grounding    # noqa: F401
    from .providers.bedrock.amazon import amazon_titan_embed_v2          # noqa: F401
except ImportError:
    pass

try:
    from .providers.gemini import gemini_3_1_flash_lite          # noqa: F401
except ImportError:
    pass

try:
    from .providers.groq import gpt_oss_120b                     # noqa: F401
    from .providers.groq import gpt_oss_20b                      # noqa: F401
    from .providers.groq import qwen_3_6_27b                     # noqa: F401
except ImportError:
    pass

__all__ = [
    "Router",
    "LLMResponse",
    "EmbeddingResponse",
    "LLMRouterError",
    "RouterValidationError",
    "ProviderNotFoundError",
    "LLMCallError",
    "help",
    "list_models",
    "model_info",
    "list_embedding_models",
    "embedding_model_info",
]
