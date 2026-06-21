"""OpenRouter API client.

OpenRouter exposes hundreds of upstream models through a single
OpenAI-compatible endpoint at https://openrouter.ai/api/v1.

Model identifiers follow the ``<vendor>/<model>`` convention
(e.g. ``openai/gpt-5.2``, ``anthropic/claude-opus-4-5``,
``meta-llama/llama-3.1-70b-instruct``). These IDs intentionally collide
with HuggingFace-style names that may also be served by a local LM
Studio / vLLM endpoint — disambiguation is the responsibility of the
routing layer in :mod:`ukrqualbench.models.routing` (``openrouter:`` vs
``local:`` prefix).

Cost tracking uses the ``usage.cost`` field returned by OpenRouter in
its normalised completion response — that is the only authoritative
source because OpenRouter prices fluctuate per upstream provider and
include OpenRouter's surcharge.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from ukrqualbench.models.base import (
    BaseModelClient,
    ModelConfig,
    ModelResponse,
)

if TYPE_CHECKING:
    import httpx


OPENROUTER_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_DEFAULT_REFERER = "https://github.com/UkrQualBench/UkrQualBench"
OPENROUTER_DEFAULT_TITLE = "UkrQualBench"


class OpenRouterClient(BaseModelClient):
    """OpenRouter chat completions client.

    Example:
        >>> client = OpenRouterClient(
        ...     model_id="openai/gpt-5.2",
        ...     api_key="sk-or-...",
        ... )
        >>> response = await client.generate("Поясніть фотосинтез")
        >>> print(response.text, response.cost_usd)

    The wrapped ``model_id`` is exposed as ``openrouter/<vendor>/<model>``
    so that downstream code (ELO registry, evaluation files) can tell
    OpenRouter-served models apart from a model of the same name running
    elsewhere (e.g. ``local/google/gemma-3-27b-it``).
    """

    def __init__(
        self,
        model_id: str,
        config: ModelConfig | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        referer: str | None = None,
        title: str | None = None,
    ) -> None:
        """Initialize OpenRouter client.

        Args:
            model_id: Upstream model identifier (``vendor/model``).
            config: Optional configuration.
            api_key: API key (defaults to OPENROUTER_API_KEY env var).
            base_url: Override OpenRouter base URL.
            referer: HTTP-Referer header for OpenRouter rankings.
            title: X-OpenRouter-Title header for OpenRouter rankings.
        """
        super().__init__(f"openrouter/{model_id}", config)
        self._upstream_model = model_id

        self._api_key = (
            api_key or self._config.api_key or os.getenv("OPENROUTER_API_KEY")
        )
        if not self._api_key:
            raise ValueError(
                "OpenRouter API key required. Set UKRQUALBENCH_OPENROUTER_API_KEY "
                "(or OPENROUTER_API_KEY) environment variable or pass api_key parameter."
            )

        self._base_url = (
            base_url
            or self._config.base_url
            or os.getenv("OPENROUTER_BASE_URL")
            or OPENROUTER_DEFAULT_BASE_URL
        )
        self._referer = referer or os.getenv(
            "OPENROUTER_HTTP_REFERER", OPENROUTER_DEFAULT_REFERER
        )
        self._title = title or os.getenv(
            "OPENROUTER_X_TITLE", OPENROUTER_DEFAULT_TITLE
        )

        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        """Return a cached httpx client with auth + attribution headers."""
        if self._client is None:
            try:
                import httpx
            except ImportError as e:
                raise ImportError(
                    "httpx package required. Install with: pip install httpx"
                ) from e

            headers: dict[str, str] = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            }
            if self._referer:
                headers["HTTP-Referer"] = self._referer
            if self._title:
                headers["X-OpenRouter-Title"] = self._title

            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._config.timeout,
                headers=headers,
            )
        return self._client

    async def _call_api(
        self,
        prompt: str,
        system_prompt: str | None,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> ModelResponse:
        """Call OpenRouter chat completions endpoint."""
        client = self._get_client()

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self._upstream_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        response = await client.post("/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()

        choices = data.get("choices") or []
        text = (
            choices[0].get("message", {}).get("content", "") if choices else ""
        ) or ""

        usage = data.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens) or 0)
        cost_raw = usage.get("cost", 0.0)
        try:
            cost = float(cost_raw) if cost_raw is not None else 0.0
        except (TypeError, ValueError):
            cost = 0.0

        return ModelResponse(
            text=text,
            tokens_used=total_tokens,
            latency_ms=0.0,
            model_id=self._model_id,
            cost_usd=cost,
        )

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


def create_openrouter_client(
    model_id: str,
    api_key: str | None = None,
    base_url: str | None = None,
    referer: str | None = None,
    title: str | None = None,
    temperature: float = 0.0,
    max_retries: int = 3,
) -> OpenRouterClient:
    """Factory for OpenRouterClient.

    Args:
        model_id: Upstream model id (``vendor/model``).
        api_key: Optional API key (overrides env var).
        base_url: Optional override of OpenRouter base URL.
        referer: Optional HTTP-Referer header.
        title: Optional X-OpenRouter-Title header.
        temperature: Default sampling temperature.
        max_retries: Maximum retry attempts.

    Returns:
        Configured OpenRouterClient instance.
    """
    config = ModelConfig(
        default_temperature=temperature,
        max_retries=max_retries,
    )
    return OpenRouterClient(
        model_id=model_id,
        config=config,
        api_key=api_key,
        base_url=base_url,
        referer=referer,
        title=title,
    )
