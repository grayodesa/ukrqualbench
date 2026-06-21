"""Model wrappers for API and local LLMs.

Provides unified interface for various LLM providers:
- OpenAI (GPT-5.2, GPT-5-mini, GPT-4o)
- Anthropic (Claude Opus 4.5, Claude Sonnet 4)
- Google (Gemini 3 Pro, Gemini 3 Flash)
- OpenRouter (unified gateway, e.g. ``openrouter:openai/gpt-5.2``)
- Local (Ollama, vLLM, LM Studio, OpenAI-compatible)

Use :func:`ukrqualbench.models.routing.create_model_client` to turn a
string ``model_id`` into the right client — it supports both the legacy
heuristic (``gpt-…``, ``claude-…``, …) and explicit ``provider:`` prefixes
(``openrouter:openai/gpt-5.2``, ``local:google/gemma-3-27b-it``).

All clients implement the ModelClient protocol from judges.base.
"""

from ukrqualbench.models.anthropic import (
    AnthropicClient,
    create_anthropic_client,
)
from ukrqualbench.models.base import (
    MODEL_PRICING,
    BaseModelClient,
    MockModelClient,
    ModelConfig,
    ModelResponse,
    calculate_cost,
)
from ukrqualbench.models.google import (
    GoogleClient,
    VertexAIClient,
    create_google_client,
)
from ukrqualbench.models.local import (
    OllamaChatClient,
    OllamaClient,
    OpenAICompatibleClient,
    VLLMClient,
    create_local_client,
    create_ollama_client,
    create_vllm_client,
)
from ukrqualbench.models.openai import (
    AzureOpenAIClient,
    OpenAIClient,
    create_openai_client,
)
from ukrqualbench.models.openrouter import (
    OpenRouterClient,
    create_openrouter_client,
)
from ukrqualbench.models.routing import (
    SUPPORTED_PROVIDERS,
    create_model_client,
    split_provider_prefix,
)

__all__ = [
    # Constants
    "MODEL_PRICING",
    "SUPPORTED_PROVIDERS",
    # Anthropic
    "AnthropicClient",
    # Azure
    "AzureOpenAIClient",
    # Base
    "BaseModelClient",
    # Google
    "GoogleClient",
    # Mock
    "MockModelClient",
    "ModelConfig",
    "ModelResponse",
    # Local
    "OllamaChatClient",
    "OllamaClient",
    # OpenAI
    "OpenAIClient",
    # OpenAI-compatible
    "OpenAICompatibleClient",
    # OpenRouter
    "OpenRouterClient",
    # vLLM
    "VLLMClient",
    # Vertex AI
    "VertexAIClient",
    # Utilities
    "calculate_cost",
    # Factory functions
    "create_anthropic_client",
    "create_google_client",
    "create_local_client",
    # Routing
    "create_model_client",
    "create_ollama_client",
    "create_openai_client",
    "create_openrouter_client",
    "create_vllm_client",
    "split_provider_prefix",
]
