"""Centralised model-client routing.

A single entry point — :func:`create_model_client` — is used by both the
CLI (``ukrqualbench.cli``) and the evaluator (``ukrqualbench.core.evaluator``)
to turn a string model identifier into a concrete API client.

Two routing modes are supported:

1. **Explicit provider prefix** — ``"<provider>:<model_id>"`` — picks the
   provider unambiguously and passes everything after the colon as the
   upstream model identifier. Recognised providers:

   ===========  ==========================================
   Prefix       Client factory
   ===========  ==========================================
   openrouter:  :func:`models.openrouter.create_openrouter_client`
   openai:      :func:`models.openai.create_openai_client`
   anthropic:   :func:`models.anthropic.create_anthropic_client`
   google:      :func:`models.google.create_google_client`
   local:       :func:`models.local.create_local_client`
   ollama:      :func:`models.local.create_ollama_client`
   vllm:        :func:`models.local.create_vllm_client`
   ===========  ==========================================

2. **Backward-compatible heuristic** — used when no explicit prefix is
   present, so existing model ids in the ELO registry, checkpoint files
   and tests keep working:

   - starts with ``gpt-``, ``o1`` or ``o3`` → OpenAI
   - starts with ``claude-`` → Anthropic
   - starts with ``gemini-`` → Google
   - contains ``gemma`` → local (LM Studio)
   - anything else → local

The heuristic intentionally routes ambiguous ``vendor/model`` ids
(e.g. ``meta-llama/llama-3.1-70b-instruct``, ``google/gemma-3-27b-it``)
to **local** — matching pre-existing behaviour. Use the explicit
``openrouter:`` prefix to override.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ukrqualbench.core.config import Config
    from ukrqualbench.models.base import BaseModelClient


# Sentinel — recognised explicit provider prefixes. Keep this exported
# so tests and CLI helpers can introspect supported providers without
# duplicating the list.
SUPPORTED_PROVIDERS: frozenset[str] = frozenset(
    {"openrouter", "openai", "anthropic", "google", "local", "ollama", "vllm"}
)


def split_provider_prefix(model_id: str) -> tuple[str | None, str]:
    """Split ``"provider:model_id"`` into ``(provider, upstream_model)``.

    Only the *first* colon is treated as a delimiter and only when the
    token before it matches a known provider. This preserves
    OpenRouter-style ids (``openai/gpt-5.2``), HuggingFace ids
    (``google/gemma-3-27b-it``) and timestamp-bearing ids untouched.

    Args:
        model_id: User-provided model identifier.

    Returns:
        ``(provider, upstream_id)`` — ``provider`` is ``None`` when no
        explicit prefix is present.

    Examples:
        >>> split_provider_prefix("openrouter:openai/gpt-5.2")
        ('openrouter', 'openai/gpt-5.2')
        >>> split_provider_prefix("gpt-5.2")
        (None, 'gpt-5.2')
        >>> split_provider_prefix("google/gemma-3-27b-it")  # ambiguous, no prefix
        (None, 'google/gemma-3-27b-it')
    """
    if ":" not in model_id:
        return None, model_id

    head, _, tail = model_id.partition(":")
    head_lower = head.lower()
    if head_lower in SUPPORTED_PROVIDERS:
        return head_lower, tail
    return None, model_id


def create_model_client(model_id: str, config: Config) -> BaseModelClient:
    """Create a model client for ``model_id`` using settings from ``config``.

    See module docstring for the routing rules.

    Args:
        model_id: Model identifier with optional ``provider:`` prefix.
        config: UkrQualBench configuration carrying API keys & endpoints.

    Returns:
        Concrete client implementing the ModelClient protocol.
    """
    # Import via the package namespace (not submodules) so test patches
    # of the form ``patch("ukrqualbench.models.create_openai_client")``
    # take effect — that is the public mock seam.
    from ukrqualbench import models as _m

    provider, upstream = split_provider_prefix(model_id)

    if provider == "openrouter":
        return _m.create_openrouter_client(
            model_id=upstream,
            api_key=config.openrouter_api_key,
            referer=config.openrouter_referer,
            title=config.openrouter_title,
            temperature=config.temperature,
            max_retries=config.max_retries,
        )
    if provider == "openai":
        return _m.create_openai_client(
            model_id=upstream,
            api_key=config.openai_api_key,
            temperature=config.temperature,
            max_retries=config.max_retries,
        )
    if provider == "anthropic":
        return _m.create_anthropic_client(
            model_id=upstream,
            api_key=config.anthropic_api_key,
            temperature=config.temperature,
            max_retries=config.max_retries,
        )
    if provider == "google":
        return _m.create_google_client(
            model_id=upstream,
            api_key=config.google_api_key,
            temperature=config.temperature,
            max_retries=config.max_retries,
        )
    if provider == "local":
        return _m.create_local_client(
            model_id=upstream,
            base_url=config.local_base_url,
            temperature=config.temperature,
        )
    if provider == "ollama":
        return _m.create_ollama_client(
            model_id=upstream,
            base_url=config.local_base_url,
            temperature=config.temperature,
        )
    if provider == "vllm":
        return _m.create_vllm_client(
            model_id=upstream,
            base_url=config.vllm_base_url,
            temperature=config.temperature,
            api_key=config.vllm_api_key,
        )

    # No explicit prefix → fall back to legacy heuristic.
    model_lower = upstream.lower()

    if "gemma" in model_lower:
        return _m.create_local_client(
            model_id=upstream,
            base_url=config.local_base_url,
            temperature=config.temperature,
        )
    if model_lower.startswith(("gpt-", "o1", "o3")):
        return _m.create_openai_client(
            model_id=upstream,
            api_key=config.openai_api_key,
            temperature=config.temperature,
            max_retries=config.max_retries,
        )
    if model_lower.startswith("claude-"):
        return _m.create_anthropic_client(
            model_id=upstream,
            api_key=config.anthropic_api_key,
            temperature=config.temperature,
            max_retries=config.max_retries,
        )
    if model_lower.startswith("gemini-"):
        return _m.create_google_client(
            model_id=upstream,
            api_key=config.google_api_key,
            temperature=config.temperature,
            max_retries=config.max_retries,
        )
    return _m.create_local_client(
        model_id=upstream,
        base_url=config.local_base_url,
        temperature=config.temperature,
    )
