"""Local and OpenAI-compatible model clients.

Supports locally running LLMs and OpenAI-compatible endpoints:
- Ollama (local)
- vLLM (local)
- LM Studio / LocalAI / llama.cpp server / generic OpenAI-compatible
"""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING, Any

from ukrqualbench.models.base import (
    BaseModelClient,
    ModelConfig,
    ModelResponse,
)

if TYPE_CHECKING:
    import httpx


def strip_thinking_tags(text: str) -> str:
    """Strip <think>...</think> reasoning blocks from model output.

    Some models (Qwen3, DeepSeek-R1) include reasoning in <think> tags
    before the actual response. This function extracts just the answer.

    Args:
        text: Raw model output potentially containing <think> blocks.

    Returns:
        Text with thinking blocks removed, stripped of whitespace.
    """
    pattern = r"<think>.*?</think>"
    result = re.sub(pattern, "", text, flags=re.DOTALL)
    return result.strip()


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
    api_key: str | None = None,
) -> VLLMClient:
    """Factory function to create a vLLM client.

    Args:
        model_id: Model identifier.
        base_url: vLLM server URL.
        temperature: Default sampling temperature.
        api_key: Optional API key for a server started with ``--api-key``.
            Falls back to the ``VLLM_API_KEY`` environment variable.

    Returns:
        Configured VLLMClient instance.
    """
    config = ModelConfig(
        base_url=base_url,
        default_temperature=temperature,
    )
    return VLLMClient(model_id=model_id, base_url=base_url, config=config, api_key=api_key)


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


