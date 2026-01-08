"""Google Gemini model client.

Supports Gemini 3 Pro, Gemini 3 Flash, and legacy Gemini 2.0/1.5 models.
"""

from __future__ import annotations

import os
from typing import Any

from ukrqualbench.models.base import (
    BaseModelClient,
    ModelConfig,
    ModelResponse,
    calculate_cost,
)


class GoogleClient(BaseModelClient):
    """Google Gemini API client.

    Supports Gemini models including:
    - gemini-3-pro-preview (flagship reasoning model)
    - gemini-3-flash-preview (fast reasoning model)
    - gemini-2.0-flash (legacy)
    - gemini-1.5-pro (legacy)

    Example:
        >>> client = GoogleClient("gemini-3-pro-preview")
        >>> response = await client.generate("Поясніть фотосинтез")
        >>> print(response.text)
    """

    def __init__(
        self,
        model_id: str = "gemini-3-pro-preview",
        config: ModelConfig | None = None,
        api_key: str | None = None,
    ) -> None:
        """Initialize Google Gemini client.

        Args:
            model_id: Gemini model identifier.
            config: Optional configuration.
            api_key: API key (defaults to GOOGLE_API_KEY env var).
        """
        super().__init__(model_id, config)

        # Resolve API key
        self._api_key = api_key or self._config.api_key or os.getenv("GOOGLE_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Google API key required. Set GOOGLE_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # Model will be created lazily
        self._model: Any = None
        self._initialized = False

    def _initialize(self) -> None:
        """Initialize the Google GenAI library."""
        if self._initialized:
            return

        try:
            import google.generativeai as genai
        except ImportError as e:
            raise ImportError(
                "google-generativeai package required. "
                "Install with: pip install google-generativeai"
            ) from e

        genai.configure(api_key=self._api_key)  # type: ignore[attr-defined]
        self._initialized = True

    def _get_model(self) -> Any:
        """Get or create the Gemini model.

        Returns:
            GenerativeModel instance.
        """
        if self._model is None:
            self._initialize()
            import google.generativeai as genai

            self._model = genai.GenerativeModel(self._model_id)  # type: ignore[attr-defined]

        return self._model

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
        import google.generativeai as genai

        self._initialize()

        # Build generation config
        generation_config = genai.GenerationConfig(  # type: ignore[attr-defined]
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        # Add JSON response format if requested
        if json_mode:
            generation_config = genai.GenerationConfig(  # type: ignore[attr-defined]
                temperature=temperature,
                max_output_tokens=max_tokens,
                response_mime_type="application/json",
            )

        # Create model with system instruction if provided
        if system_prompt:
            model = genai.GenerativeModel(  # type: ignore[attr-defined]
                self._model_id,
                system_instruction=system_prompt,
                generation_config=generation_config,
            )
        else:
            model = genai.GenerativeModel(  # type: ignore[attr-defined]
                self._model_id,
                generation_config=generation_config,
            )

        # Generate response
        response = await model.generate_content_async(prompt)

        # Extract response content
        text = response.text if response.text else ""

        # Get token counts from usage metadata
        input_tokens = 0
        output_tokens = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0)
            output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0)

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

    Example:
        >>> client = VertexAIClient(
        ...     model_id="gemini-3-pro-preview",
        ...     project_id="my-project",
        ...     location="us-central1",
        ... )
        >>> response = await client.generate("Поясніть фотосинтез")
    """

    def __init__(
        self,
        model_id: str = "gemini-3-pro-preview",
        project_id: str | None = None,
        location: str = "us-central1",
        config: ModelConfig | None = None,
    ) -> None:
        """Initialize Vertex AI client.

        Args:
            model_id: Gemini model identifier.
            project_id: GCP project ID (defaults to GOOGLE_CLOUD_PROJECT env var).
            location: GCP region.
            config: Optional configuration.
        """
        super().__init__(model_id, config)

        self._project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        if not self._project_id:
            raise ValueError(
                "Google Cloud project ID required. Set GOOGLE_CLOUD_PROJECT "
                "environment variable or pass project_id parameter."
            )

        self._location = location
        self._model: Any = None
        self._initialized = False

    def _initialize(self) -> None:
        """Initialize Vertex AI."""
        if self._initialized:
            return

        try:
            import vertexai  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "google-cloud-aiplatform package required. "
                "Install with: pip install google-cloud-aiplatform"
            ) from e

        vertexai.init(project=self._project_id, location=self._location)
        self._initialized = True

    def _get_model(self) -> Any:
        """Get or create the Vertex AI model.

        Returns:
            GenerativeModel instance.
        """
        if self._model is None:
            self._initialize()
            from vertexai.generative_models import GenerativeModel  # type: ignore[import-not-found]

            self._model = GenerativeModel(self._model_id)

        return self._model

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
        self._initialize()
        from vertexai.generative_models import GenerationConfig, GenerativeModel

        # Build generation config
        gen_config_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }

        if json_mode:
            gen_config_kwargs["response_mime_type"] = "application/json"

        generation_config = GenerationConfig(**gen_config_kwargs)

        # Create model with system instruction if provided
        if system_prompt:
            model = GenerativeModel(
                self._model_id,
                system_instruction=system_prompt,
            )
        else:
            model = self._get_model()

        # Generate response (Vertex AI doesn't have native async, run in executor)
        import asyncio

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: model.generate_content(prompt, generation_config=generation_config),
        )

        # Extract response content
        text = response.text if response.text else ""

        # Get token counts
        input_tokens = 0
        output_tokens = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0)
            output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0)

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
    model_id: str = "gemini-3-pro-preview",
    api_key: str | None = None,
    temperature: float = 0.0,
    max_retries: int = 3,
) -> GoogleClient:
    """Factory function to create a Google Gemini client.

    Args:
        model_id: Gemini model identifier (default: gemini-3-pro-preview).
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
