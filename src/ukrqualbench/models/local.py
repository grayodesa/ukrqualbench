"""Local and OpenAI-compatible model clients.

Supports locally running LLMs and OpenAI-compatible cloud providers:
- Ollama (local)
- vLLM (local)
- Nebius Token Factory (cloud)
- Generic OpenAI-compatible endpoints
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
    import httpx


class OllamaClient(BaseModelClient):
    """Ollama API client.

    Supports any model available through Ollama including:
    - llama3.2, llama3.1
    - mistral, mixtral
    - qwen2.5
    - phi3
    - Custom fine-tuned models

    Example:
        >>> client = OllamaClient("llama3.2")
        >>> response = await client.generate("Поясніть фотосинтез")
        >>> print(response.text)
    """

    def __init__(
        self,
        model_id: str = "llama3.2",
        config: ModelConfig | None = None,
        base_url: str | None = None,
    ) -> None:
        """Initialize Ollama client.

        Args:
            model_id: Ollama model name.
            config: Optional configuration.
            base_url: Ollama server URL (defaults to OLLAMA_HOST or http://localhost:11434).
        """
        super().__init__(f"ollama/{model_id}", config)
        self._ollama_model = model_id

        # Resolve base URL (check both OLLAMA_BASE_URL and legacy OLLAMA_HOST)
        self._base_url = (
            base_url
            or self._config.base_url
            or os.getenv("OLLAMA_BASE_URL")
            or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        )

        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client.

        Returns:
            AsyncClient instance.
        """
        if self._client is None:
            try:
                import httpx
            except ImportError as e:
                raise ImportError("httpx package required. Install with: pip install httpx") from e

            assert self._base_url is not None  # Always set via default
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
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
        """Make Ollama API call.

        Args:
            prompt: User prompt.
            system_prompt: Optional system prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            json_mode: Whether to enforce JSON output.

        Returns:
            Model response.

        Raises:
            httpx.HTTPError: If API call fails.
        """
        client = self._get_client()

        # Build request payload
        payload: dict[str, Any] = {
            "model": self._ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        if system_prompt:
            payload["system"] = system_prompt

        if json_mode:
            payload["format"] = "json"

        response = await client.post("/api/generate", json=payload)
        response.raise_for_status()

        data = response.json()

        # Extract response content
        text = data.get("response", "")

        # Get token counts (Ollama provides these)
        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)
        total_tokens = prompt_tokens + completion_tokens

        return ModelResponse(
            text=text,
            tokens_used=total_tokens,
            latency_ms=0.0,  # Will be recalculated by base class
            model_id=self._model_id,
            cost_usd=0.0,  # Local models are free
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


class OllamaChatClient(BaseModelClient):
    """Ollama chat API client.

    Uses the /api/chat endpoint for multi-turn conversations.

    Example:
        >>> client = OllamaChatClient("llama3.2")
        >>> response = await client.generate("Поясніть фотосинтез")
        >>> print(response.text)
    """

    def __init__(
        self,
        model_id: str = "llama3.2",
        config: ModelConfig | None = None,
        base_url: str | None = None,
    ) -> None:
        """Initialize Ollama chat client.

        Args:
            model_id: Ollama model name.
            config: Optional configuration.
            base_url: Ollama server URL.
        """
        super().__init__(f"ollama/{model_id}", config)
        self._ollama_model = model_id

        # Resolve base URL (check both OLLAMA_BASE_URL and legacy OLLAMA_HOST)
        self._base_url = (
            base_url
            or self._config.base_url
            or os.getenv("OLLAMA_BASE_URL")
            or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        )

        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            import httpx

            assert self._base_url is not None  # Always set via default
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
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
        """Make Ollama chat API call.

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

        # Build messages
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self._ollama_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        if json_mode:
            payload["format"] = "json"

        response = await client.post("/api/chat", json=payload)
        response.raise_for_status()

        data = response.json()

        # Extract response content
        text = data.get("message", {}).get("content", "")

        # Get token counts
        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)
        total_tokens = prompt_tokens + completion_tokens

        return ModelResponse(
            text=text,
            tokens_used=total_tokens,
            latency_ms=0.0,
            model_id=self._model_id,
            cost_usd=0.0,
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


class VLLMClient(BaseModelClient):
    """vLLM OpenAI-compatible API client.

    Supports models served through vLLM with OpenAI-compatible API.

    Example:
        >>> client = VLLMClient(
        ...     model_id="meta-llama/Llama-3.2-3B-Instruct",
        ...     base_url="http://localhost:8000",
        ... )
        >>> response = await client.generate("Поясніть фотосинтез")
        >>> print(response.text)
    """

    def __init__(
        self,
        model_id: str,
        base_url: str = "http://localhost:8000",
        config: ModelConfig | None = None,
        api_key: str | None = None,
    ) -> None:
        """Initialize vLLM client.

        Args:
            model_id: Model identifier (as registered in vLLM).
            base_url: vLLM server URL.
            config: Optional configuration.
            api_key: Optional API key if vLLM requires auth.
        """
        super().__init__(f"vllm/{model_id}", config)
        self._vllm_model = model_id
        self._base_url = base_url or self._config.base_url or "http://localhost:8000"
        self._api_key = api_key or self._config.api_key or os.getenv("VLLM_API_KEY", "")

        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            import httpx

            headers = {}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"

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
        """Make vLLM OpenAI-compatible API call.

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

        # Build messages (OpenAI format)
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self._vllm_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        response = await client.post("/v1/chat/completions", json=payload)
        response.raise_for_status()

        data = response.json()

        # Extract response content (OpenAI format)
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")

        # Get token counts
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = prompt_tokens + completion_tokens

        return ModelResponse(
            text=text,
            tokens_used=total_tokens,
            latency_ms=0.0,
            model_id=self._model_id,
            cost_usd=0.0,  # Local models are free
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


class NebiusClient(BaseModelClient):
    """Nebius Token Factory API client.

    Supports OpenAI-compatible API for cloud-hosted open-source models:
    - DeepSeek-R1-0528, DeepSeek-V3
    - Qwen3-235B, Qwen3-30B, QwQ-32B
    - Meta/Llama-3.3-70B, Meta/Llama-3.1-8B, Meta/Llama-3.1-405B
    - gpt-oss-120b, gpt-oss-20b
    - And more

    See https://docs.tokenfactory.nebius.com for full model list.

    Example:
        >>> client = NebiusClient("deepseek-ai/DeepSeek-R1-0528")
        >>> response = await client.generate("Поясніть фотосинтез")
        >>> print(response.text)
    """

    # Default API endpoint
    DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1"

    def __init__(
        self,
        model_id: str = "deepseek-ai/DeepSeek-R1-0528",
        config: ModelConfig | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        """Initialize Nebius Token Factory client.

        Args:
            model_id: Model identifier (e.g., "deepseek-ai/DeepSeek-R1-0528").
            config: Optional configuration.
            api_key: API key (defaults to NEBIUS_API_KEY env var).
            base_url: Optional custom base URL.
        """
        super().__init__(f"nebius/{model_id}", config)
        self._nebius_model = model_id

        # Resolve API key
        self._api_key = api_key or self._config.api_key or os.getenv("NEBIUS_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Nebius API key required. Set UKRQUALBENCH_NEBIUS_API_KEY (or NEBIUS_API_KEY) "
                "environment variable or pass api_key parameter. "
                "Get your key at https://tokenfactory.nebius.com/"
            )

        # Resolve base URL
        self._base_url = base_url or self._config.base_url or self.DEFAULT_BASE_URL

        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._config.timeout,
                headers={"Authorization": f"Bearer {self._api_key}"},
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
        """Make Nebius Token Factory API call.

        Args:
            prompt: User prompt.
            system_prompt: Optional system prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            json_mode: Whether to enforce JSON output.

        Returns:
            Model response.

        Raises:
            httpx.HTTPError: If API call fails.
        """
        client = self._get_client()

        # Build messages (OpenAI format)
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self._nebius_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        response = await client.post("/chat/completions", json=payload)
        response.raise_for_status()

        data = response.json()

        # Extract response content (OpenAI format)
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")

        # Get token counts
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = prompt_tokens + completion_tokens

        # Calculate cost using pricing from base.py
        cost = calculate_cost(self._model_id, prompt_tokens, completion_tokens)

        return ModelResponse(
            text=text,
            tokens_used=total_tokens,
            latency_ms=0.0,
            model_id=self._model_id,
            cost_usd=cost,
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


class OpenAICompatibleClient(BaseModelClient):
    """Generic OpenAI-compatible API client.

    Supports any server that implements the OpenAI chat completions API:
    - LM Studio
    - LocalAI
    - Text Generation Inference
    - llama.cpp server

    Example:
        >>> client = OpenAICompatibleClient(
        ...     model_id="local-model",
        ...     base_url="http://localhost:1234/v1",
        ... )
        >>> response = await client.generate("Поясніть фотосинтез")
    """

    def __init__(
        self,
        model_id: str,
        base_url: str,
        config: ModelConfig | None = None,
        api_key: str | None = None,
    ) -> None:
        """Initialize OpenAI-compatible client.

        Args:
            model_id: Model identifier.
            base_url: Server base URL (should include /v1 if needed).
            config: Optional configuration.
            api_key: Optional API key.
        """
        super().__init__(f"local/{model_id}", config)
        self._local_model = model_id
        self._base_url = base_url
        self._api_key = api_key or self._config.api_key

        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            import httpx

            headers = {}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"

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
        """Make OpenAI-compatible API call.

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

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self._local_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        response = await client.post("/chat/completions", json=payload)
        response.raise_for_status()

        data = response.json()

        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")

        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = prompt_tokens + completion_tokens

        return ModelResponse(
            text=text,
            tokens_used=total_tokens,
            latency_ms=0.0,
            model_id=self._model_id,
            cost_usd=0.0,
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


def create_ollama_client(
    model_id: str = "llama3.2",
    base_url: str | None = None,
    temperature: float = 0.0,
    use_chat_api: bool = True,
) -> OllamaClient | OllamaChatClient:
    """Factory function to create an Ollama client.

    Args:
        model_id: Ollama model name.
        base_url: Optional server URL.
        temperature: Default sampling temperature.
        use_chat_api: Whether to use chat API (recommended).

    Returns:
        Configured Ollama client instance.
    """
    config = ModelConfig(
        base_url=base_url,
        default_temperature=temperature,
    )

    if use_chat_api:
        return OllamaChatClient(model_id=model_id, config=config)
    return OllamaClient(model_id=model_id, config=config)


def create_vllm_client(
    model_id: str,
    base_url: str = "http://localhost:8000",
    temperature: float = 0.0,
) -> VLLMClient:
    """Factory function to create a vLLM client.

    Args:
        model_id: Model identifier.
        base_url: vLLM server URL.
        temperature: Default sampling temperature.

    Returns:
        Configured VLLMClient instance.
    """
    config = ModelConfig(
        base_url=base_url,
        default_temperature=temperature,
    )
    return VLLMClient(model_id=model_id, base_url=base_url, config=config)


def create_local_client(
    model_id: str,
    base_url: str | None = None,
    temperature: float = 0.0,
) -> OpenAICompatibleClient:
    """Factory function to create a local OpenAI-compatible client.

    Works with LM Studio, Ollama (OpenAI mode), vLLM, LocalAI, llama.cpp server, etc.

    Args:
        model_id: Model identifier (as shown in your local server).
        base_url: Server URL (defaults to UKRQUALBENCH_OLLAMA_BASE_URL or http://localhost:1234/v1).
        temperature: Default sampling temperature.

    Returns:
        Configured OpenAICompatibleClient instance.
    """
    import os

    resolved_url = (
        base_url
        or os.getenv("UKRQUALBENCH_OLLAMA_BASE_URL")
        or os.getenv("UKRQUALBENCH_LOCAL_BASE_URL")
        or "http://localhost:1234/v1"
    )
    config = ModelConfig(default_temperature=temperature)
    return OpenAICompatibleClient(model_id=model_id, base_url=resolved_url, config=config)


def create_nebius_client(
    model_id: str = "deepseek-ai/DeepSeek-R1-0528",
    api_key: str | None = None,
    temperature: float = 0.0,
    max_retries: int = 3,
) -> NebiusClient:
    """Factory function to create a Nebius Token Factory client.

    Args:
        model_id: Model identifier (e.g., "deepseek-ai/DeepSeek-R1-0528").
        api_key: Optional API key (defaults to NEBIUS_API_KEY env var).
        temperature: Default sampling temperature.
        max_retries: Maximum retry attempts.

    Returns:
        Configured NebiusClient instance.

    Available models include:
        - deepseek-ai/DeepSeek-R1-0528 (reasoning)
        - deepseek-ai/DeepSeek-V3 (general)
        - Qwen/Qwen3-235B-Instruct
        - Qwen/Qwen3-30B-Instruct
        - meta-llama/Llama-3.3-70B-Instruct
        - meta-llama/Llama-3.1-405B-Instruct
        - gpt-oss-120b

    See https://docs.tokenfactory.nebius.com for full model list.
    """
    config = ModelConfig(
        api_key=api_key,
        default_temperature=temperature,
        max_retries=max_retries,
    )
    return NebiusClient(model_id=model_id, config=config)
