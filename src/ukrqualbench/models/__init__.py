"""Model wrappers for API and local LLMs.

Provides unified interface for various LLM providers:
- OpenAI (GPT-5.2, GPT-5-mini, GPT-4o)
- Anthropic (Claude Opus 4.5, Claude Sonnet 4)
- Google (Gemini 3 Pro, Gemini 3 Flash)
- Nebius Token Factory (DeepSeek, Qwen, Llama)
- Local (Ollama, vLLM, OpenAI-compatible)

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
    NebiusClient,
    OllamaChatClient,
    OllamaClient,
    OpenAICompatibleClient,
    VLLMClient,
    create_local_client,
    create_nebius_client,
    create_ollama_client,
    create_vllm_client,
)
from ukrqualbench.models.openai import (
    AzureOpenAIClient,
    OpenAIClient,
    create_openai_client,
)

__all__ = [
    # Constants
    "MODEL_PRICING",
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
    # Nebius Token Factory
    "NebiusClient",
    # Local
    "OllamaChatClient",
    "OllamaClient",
    # OpenAI
    "OpenAIClient",
    # OpenAI-compatible
    "OpenAICompatibleClient",
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
    "create_nebius_client",
    "create_ollama_client",
    "create_openai_client",
    "create_vllm_client",
]
