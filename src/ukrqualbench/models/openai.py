"""OpenAI model client.

Supports GPT-5.2, GPT-5-mini, GPT-4o and Azure OpenAI endpoints.
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
    import openai


class OpenAIClient(BaseModelClient):
    """OpenAI API client.

    Supports all OpenAI chat models including:
    - gpt-5.2, gpt-5.2-pro (flagship models)
    - gpt-5-mini (efficient model)
    - gpt-5.1-codex-max (coding specialist)
    - gpt-4o, gpt-4o-mini (legacy)

    Example:
        >>> client = OpenAIClient("gpt-5.2")
        >>> response = await client.generate("Поясніть фотосинтез")
        >>> print(response.text)
    """

    def __init__(
        self,
        model_id: str = "gpt-5.2",
        config: ModelConfig | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        """Initialize OpenAI client.

        Args:
            model_id: OpenAI model identifier.
            config: Optional configuration.
            api_key: API key (defaults to OPENAI_API_KEY env var).
            base_url: Optional base URL for Azure or custom endpoints.
        """
        super().__init__(model_id, config)

        # Resolve API key
        self._api_key = api_key or self._config.api_key or os.getenv("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # Resolve base URL
        self._base_url = base_url or self._config.base_url or os.getenv("OPENAI_BASE_URL")

        # Client will be created lazily
        self._client: openai.AsyncOpenAI | None = None

    def _get_client(self) -> openai.AsyncOpenAI:
        """Get or create the OpenAI client.

        Returns:
            AsyncOpenAI client instance.
        """
        if self._client is None:
            try:
                import openai
            except ImportError as e:
                raise ImportError(
                    "openai package required. Install with: pip install openai"
                ) from e

            client_kwargs: dict[str, Any] = {
                "api_key": self._api_key,
                "timeout": self._config.timeout,
            }
            if self._base_url:
                client_kwargs["base_url"] = self._base_url

            self._client = openai.AsyncOpenAI(**client_kwargs)

        return self._client

    async def _call_api(
        self,
        prompt: str,
        system_prompt: str | None,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> ModelResponse:
        """Make OpenAI API call.

        Args:
            prompt: User prompt.
            system_prompt: Optional system prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            json_mode: Whether to enforce JSON output.

        Returns:
            Model response.

        Raises:
            openai.APIError: If API call fails.
        """
        client = self._get_client()
        messages = self._build_messages(prompt, system_prompt)

        # Build request kwargs
        request_kwargs: dict[str, Any] = {
            "model": self._model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Add response format for JSON mode
        if json_mode:
            request_kwargs["response_format"] = {"type": "json_object"}

        response = await client.chat.completions.create(**request_kwargs)

        # Extract response content
        text = response.choices[0].message.content or ""
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0
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


class AzureOpenAIClient(BaseModelClient):
    """Azure OpenAI API client.

    Supports Azure-hosted OpenAI models.

    Example:
        >>> client = AzureOpenAIClient(
        ...     deployment_name="gpt-4o-deployment",
        ...     api_version="2024-02-15-preview",
        ... )
        >>> response = await client.generate("Поясніть фотосинтез")
    """

    def __init__(
        self,
        deployment_name: str,
        api_version: str = "2024-02-15-preview",
        config: ModelConfig | None = None,
        api_key: str | None = None,
        azure_endpoint: str | None = None,
    ) -> None:
        """Initialize Azure OpenAI client.

        Args:
            deployment_name: Azure deployment name.
            api_version: Azure API version.
            config: Optional configuration.
            api_key: API key (defaults to AZURE_OPENAI_API_KEY env var).
            azure_endpoint: Azure endpoint URL (defaults to AZURE_OPENAI_ENDPOINT env var).
        """
        super().__init__(deployment_name, config)
        self._deployment_name = deployment_name
        self._api_version = api_version

        # Resolve API key
        self._api_key = api_key or self._config.api_key or os.getenv("AZURE_OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Azure OpenAI API key required. Set AZURE_OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # Resolve endpoint
        self._azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        if not self._azure_endpoint:
            raise ValueError(
                "Azure OpenAI endpoint required. Set AZURE_OPENAI_ENDPOINT environment variable "
                "or pass azure_endpoint parameter."
            )

        self._client: openai.AsyncAzureOpenAI | None = None

    def _get_client(self) -> openai.AsyncAzureOpenAI:
        """Get or create the Azure OpenAI client.

        Returns:
            AsyncAzureOpenAI client instance.
        """
        if self._client is None:
            try:
                import openai
            except ImportError as e:
                raise ImportError(
                    "openai package required. Install with: pip install openai"
                ) from e

            assert self._azure_endpoint is not None  # Validated in __init__
            self._client = openai.AsyncAzureOpenAI(
                api_key=self._api_key,
                api_version=self._api_version,
                azure_endpoint=self._azure_endpoint,
                timeout=self._config.timeout,
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
        """Make Azure OpenAI API call.

        Args:
            prompt: User prompt.
            system_prompt: Optional system prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            json_mode: Whether to enforce JSON output.

        Returns:
            Model response.
        """
        client = self._get_client()
        messages = self._build_messages(prompt, system_prompt)

        request_kwargs: dict[str, Any] = {
            "model": self._deployment_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if json_mode:
            request_kwargs["response_format"] = {"type": "json_object"}

        response = await client.chat.completions.create(**request_kwargs)

        text = response.choices[0].message.content or ""
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0
        total_tokens = input_tokens + output_tokens

        # Azure pricing varies by deployment, use 0 as placeholder
        cost = 0.0

        return ModelResponse(
            text=text,
            tokens_used=total_tokens,
            latency_ms=0.0,
            model_id=self._deployment_name,
            cost_usd=cost,
        )


def create_openai_client(
    model_id: str = "gpt-5.2",
    api_key: str | None = None,
    temperature: float = 0.0,
    max_retries: int = 3,
) -> OpenAIClient:
    """Factory function to create an OpenAI client.

    Args:
        model_id: OpenAI model identifier (default: gpt-5.2).
        api_key: Optional API key.
        temperature: Default sampling temperature.
        max_retries: Maximum retry attempts.

    Returns:
        Configured OpenAIClient instance.
    """
    config = ModelConfig(
        api_key=api_key,
        default_temperature=temperature,
        max_retries=max_retries,
    )
    return OpenAIClient(model_id=model_id, config=config)
