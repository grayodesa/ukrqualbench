"""Anthropic model client.

Supports Claude Opus 4.5, Claude Sonnet 4, Claude Haiku 4, and legacy Claude 3.5 models.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from ukrqualbench.models.base import (
    BaseModelClient,
    ModelConfig,
    ModelResponse,
    calculate_cost,
)

if TYPE_CHECKING:
    import anthropic


class AnthropicClient(BaseModelClient):
    """Anthropic API client.

    Supports all Claude models including:
    - claude-opus-4-5-20251101 (flagship, most intelligent)
    - claude-sonnet-4-20250514 (balanced performance)
    - claude-haiku-4 (fast and efficient)
    - claude-3-5-sonnet-20241022 (legacy)
    - claude-3-5-haiku-20241022 (legacy)

    Example:
        >>> client = AnthropicClient("claude-opus-4-5-20251101")
        >>> response = await client.generate("Поясніть фотосинтез")
        >>> print(response.text)
    """

    def __init__(
        self,
        model_id: str = "claude-opus-4-5-20251101",
        config: ModelConfig | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        """Initialize Anthropic client.

        Args:
            model_id: Anthropic model identifier.
            config: Optional configuration.
            api_key: API key (defaults to ANTHROPIC_API_KEY env var).
            base_url: Optional base URL for custom endpoints.
        """
        super().__init__(model_id, config)

        # Resolve API key
        self._api_key = api_key or self._config.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Anthropic API key required. Set UKRQUALBENCH_ANTHROPIC_API_KEY (or ANTHROPIC_API_KEY) "
                "environment variable or pass api_key parameter."
            )

        # Resolve base URL
        self._base_url = base_url or self._config.base_url

        # Client will be created lazily
        self._client: anthropic.AsyncAnthropic | None = None

    def _get_client(self) -> anthropic.AsyncAnthropic:
        """Get or create the Anthropic client.

        Returns:
            AsyncAnthropic client instance.
        """
        if self._client is None:
            try:
                import anthropic
            except ImportError as e:
                raise ImportError(
                    "anthropic package required. Install with: pip install anthropic"
                ) from e

            client_kwargs: dict[str, Any] = {
                "api_key": self._api_key,
                "timeout": self._config.timeout,
            }
            if self._base_url:
                client_kwargs["base_url"] = self._base_url

            self._client = anthropic.AsyncAnthropic(**client_kwargs)

        return self._client

    async def _call_api(
        self,
        prompt: str,
        system_prompt: str | None,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> ModelResponse:
        """Make Anthropic API call.

        Args:
            prompt: User prompt.
            system_prompt: Optional system prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            json_mode: Whether to enforce JSON output (appends instruction).

        Returns:
            Model response.

        Raises:
            anthropic.APIError: If API call fails.
        """
        client = self._get_client()

        # Build messages - Anthropic uses different format
        messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]

        # Modify prompt for JSON mode (Anthropic doesn't have native JSON mode)
        actual_prompt = prompt
        if json_mode:
            actual_prompt = f"{prompt}\n\nRespond with valid JSON only."
            messages = [{"role": "user", "content": actual_prompt}]

        # Build request kwargs
        request_kwargs: dict[str, Any] = {
            "model": self._model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Add system prompt if provided
        if system_prompt:
            request_kwargs["system"] = system_prompt

        response = await client.messages.create(**request_kwargs)

        # Extract response content
        text = ""
        for block in response.content:
            if block.type == "text":
                text += block.text

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        total_tokens = input_tokens + output_tokens

        # Calculate cost
        cost = calculate_cost(self._model_id, input_tokens, output_tokens)

        return ModelResponse(
            text=text,
            tokens_used=total_tokens,
            latency_ms=0.0,  # Will be recalculated by base class
            model_id=self._model_id,
            cost_usd=cost,
        )


def create_anthropic_client(
    model_id: str = "claude-opus-4-5-20251101",
    api_key: str | None = None,
    temperature: float = 0.0,
    max_retries: int = 3,
) -> AnthropicClient:
    """Factory function to create an Anthropic client.

    Args:
        model_id: Anthropic model identifier (default: claude-opus-4-5-20251101).
        api_key: Optional API key.
        temperature: Default sampling temperature.
        max_retries: Maximum retry attempts.

    Returns:
        Configured AnthropicClient instance.
    """
    config = ModelConfig(
        api_key=api_key,
        default_temperature=temperature,
        max_retries=max_retries,
    )
    return AnthropicClient(model_id=model_id, config=config)
