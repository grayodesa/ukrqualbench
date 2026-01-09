"""Google Gemini model client.

Supports Gemini 2.5/2.0 models via both AI Studio and Vertex AI.
Uses the unified google-genai SDK (replaces deprecated google-generativeai).
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
    from google import genai


class GoogleClient(BaseModelClient):
    """Google Gemini API client using AI Studio.

    Supports Gemini models including:
    - gemini-2.5-pro, gemini-3-flash-preview (latest)
    - gemini-2.0-flash (stable)
    - gemini-1.5-pro, gemini-1.5-flash (legacy)

    Example:
        >>> client = GoogleClient("gemini-3-flash-preview")
        >>> response = await client.generate("Поясніть фотосинтез")
        >>> print(response.text)
    """

    def __init__(
        self,
        model_id: str = "gemini-3-flash-preview",
        config: ModelConfig | None = None,
        api_key: str | None = None,
    ) -> None:
        """Initialize Google Gemini client.

        Args:
            model_id: Gemini model identifier.
            config: Optional configuration.
            api_key: API key (defaults to GEMINI_API_KEY or GOOGLE_API_KEY env var).
        """
        super().__init__(model_id, config)

        # Resolve API key (check both GEMINI_API_KEY and legacy GOOGLE_API_KEY)
        self._api_key = (
            api_key
            or self._config.api_key
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
        )
        if not self._api_key:
            raise ValueError(
                "Google API key required. Set UKRQUALBENCH_GOOGLE_API_KEY, GEMINI_API_KEY, "
                "or GOOGLE_API_KEY environment variable, or pass api_key parameter."
            )

        # Client will be created lazily
        self._client: genai.Client | None = None

    def _get_client(self) -> genai.Client:
        """Get or create the Gemini client.

        Returns:
            genai.Client instance.
        """
        if self._client is None:
            try:
                from google import genai
            except ImportError as e:
                raise ImportError(
                    "google-genai package required. "
                    "Install with: pip install google-genai"
                ) from e

            self._client = genai.Client(api_key=self._api_key)

        return self._client

    async def _call_api(
        self,
        prompt: str,
        system_prompt: str | None,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> ModelResponse:
        """Make Google Gemini API call.

        Args:
            prompt: User prompt.
            system_prompt: Optional system prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            json_mode: Whether to enforce JSON output.

        Returns:
            Model response.

        Raises:
            google.api_core.exceptions.GoogleAPIError: If API call fails.
        """
        from google.genai import types

        client = self._get_client()

        # Build generation config
        config_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }

        # Add system instruction if provided
        if system_prompt:
            config_kwargs["system_instruction"] = system_prompt

        # Add JSON response format if requested
        if json_mode:
            config_kwargs["response_mime_type"] = "application/json"

        generation_config = types.GenerateContentConfig(**config_kwargs)

        # Generate response using async client
        response = await client.aio.models.generate_content(
            model=self._model_id,
            contents=prompt,
            config=generation_config,
        )

        # Extract response content
        text = response.text if response.text else ""

        # Get token counts from usage metadata
        input_tokens = 0
        output_tokens = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
            output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0

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


class VertexAIClient(BaseModelClient):
    """Google Vertex AI client.

    Supports Gemini models via Vertex AI for enterprise use.
    Uses the unified google-genai SDK with vertexai=True.

    Example:
        >>> client = VertexAIClient(
        ...     model_id="gemini-3-flash-preview",
        ...     project_id="my-project",
        ...     location="us-central1",
        ... )
        >>> response = await client.generate("Поясніть фотосинтез")
    """

    def __init__(
        self,
        model_id: str = "gemini-3-flash-preview",
        project_id: str | None = None,
        location: str = "us-central1",
        config: ModelConfig | None = None,
    ) -> None:
        """Initialize Vertex AI client.

        Args:
            model_id: Gemini model identifier.
            project_id: GCP project ID (defaults to GOOGLE_CLOUD_PROJECT env var).
            location: GCP region (defaults to us-central1).
            config: Optional configuration.
        """
        super().__init__(model_id, config)

        self._project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        if not self._project_id:
            raise ValueError(
                "Google Cloud project ID required. Set UKRQUALBENCH_GOOGLE_CLOUD_PROJECT "
                "(or GOOGLE_CLOUD_PROJECT) environment variable or pass project_id parameter."
            )

        self._location = location
        self._client: genai.Client | None = None

    def _get_client(self) -> genai.Client:
        """Get or create the Vertex AI client.

        Returns:
            genai.Client instance configured for Vertex AI.
        """
        if self._client is None:
            try:
                from google import genai
            except ImportError as e:
                raise ImportError(
                    "google-genai package required. "
                    "Install with: pip install google-genai"
                ) from e

            self._client = genai.Client(
                vertexai=True,
                project=self._project_id,
                location=self._location,
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
        """Make Vertex AI API call.

        Args:
            prompt: User prompt.
            system_prompt: Optional system prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            json_mode: Whether to enforce JSON output.

        Returns:
            Model response.
        """
        from google.genai import types

        client = self._get_client()

        # Build generation config
        config_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }

        # Add system instruction if provided
        if system_prompt:
            config_kwargs["system_instruction"] = system_prompt

        # Add JSON response format if requested
        if json_mode:
            config_kwargs["response_mime_type"] = "application/json"

        generation_config = types.GenerateContentConfig(**config_kwargs)

        # Generate response using async client
        response = await client.aio.models.generate_content(
            model=self._model_id,
            contents=prompt,
            config=generation_config,
        )

        # Extract response content
        text = response.text if response.text else ""

        # Get token counts
        input_tokens = 0
        output_tokens = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
            output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0

        total_tokens = input_tokens + output_tokens

        # Calculate cost (Vertex AI pricing)
        cost = calculate_cost(self._model_id, input_tokens, output_tokens)

        return ModelResponse(
            text=text,
            tokens_used=total_tokens,
            latency_ms=0.0,
            model_id=self._model_id,
            cost_usd=cost,
        )


def create_google_client(
    model_id: str = "gemini-3-flash-preview",
    api_key: str | None = None,
    temperature: float = 0.0,
    max_retries: int = 3,
) -> GoogleClient:
    """Factory function to create a Google Gemini client.

    Args:
        model_id: Gemini model identifier (default: gemini-3-flash-preview).
        api_key: Optional API key.
        temperature: Default sampling temperature.
        max_retries: Maximum retry attempts.

    Returns:
        Configured GoogleClient instance.
    """
    config = ModelConfig(
        api_key=api_key,
        default_temperature=temperature,
        max_retries=max_retries,
    )
    return GoogleClient(model_id=model_id, config=config)


def create_vertex_client(
    model_id: str = "gemini-3-flash-preview",
    project_id: str | None = None,
    location: str = "us-central1",
    temperature: float = 0.0,
    max_retries: int = 3,
) -> VertexAIClient:
    """Factory function to create a Vertex AI client.

    Args:
        model_id: Gemini model identifier (default: gemini-3-flash-preview).
        project_id: GCP project ID.
        location: GCP region.
        temperature: Default sampling temperature.
        max_retries: Maximum retry attempts.

    Returns:
        Configured VertexAIClient instance.
    """
    config = ModelConfig(
        default_temperature=temperature,
        max_retries=max_retries,
    )
    return VertexAIClient(
        model_id=model_id,
        project_id=project_id,
        location=location,
        config=config,
    )
