"""Base model client with shared functionality.

Provides abstract base class and utilities for LLM API clients.
All model clients must implement the ModelClient protocol from judges.base.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass
class ModelResponse:
    """Response from a model client.

    Duplicated from judges.base to avoid circular imports.
    Both definitions are compatible.
    """

    text: str
    tokens_used: int
    latency_ms: float
    model_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    cost_usd: float = 0.0


@dataclass
class ModelConfig:
    """Configuration for model clients."""

    api_key: str | None = None
    base_url: str | None = None
    timeout: float = 60.0
    max_retries: int = 3
    retry_delay: float = 1.0
    default_temperature: float = 0.0
    default_max_tokens: int = 1024


# Pricing per 1M tokens (input, output) as of January 2026
MODEL_PRICING: dict[str, tuple[float, float]] = {
    # OpenAI GPT-5 family
    "gpt-5.2": (5.00, 15.00),
    "gpt-5.2-pro": (10.00, 30.00),
    "gpt-5.1": (3.00, 12.00),
    "gpt-5-mini": (0.50, 1.50),
    "gpt-5.1-codex-max": (5.00, 15.00),
    # OpenAI legacy (still available)
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    # Anthropic Claude 4 family
    "claude-opus-4-5-20251101": (5.00, 25.00),
    "claude-opus-4-5": (5.00, 25.00),
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-sonnet-4": (3.00, 15.00),
    "claude-haiku-4": (0.80, 4.00),
    # Anthropic legacy
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    "claude-3-5-haiku-20241022": (0.80, 4.00),
    # Google Gemini 3 family
    "gemini-3-pro-preview": (2.50, 10.00),
    "gemini-3-flash-preview": (0.15, 0.60),
    # Google legacy
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-1.5-pro": (1.25, 5.00),
    "gemini-1.5-flash": (0.075, 0.30),
    # Nebius Token Factory - DeepSeek models
    "nebius/deepseek-ai/DeepSeek-R1-0528": (0.80, 2.40),
    "nebius/deepseek-ai/DeepSeek-V3": (0.50, 1.50),
    # Nebius Token Factory - Qwen models
    "nebius/Qwen/Qwen3-Coder-480B": (0.40, 1.80),
    "nebius/Qwen/Qwen3-235B-Thinking": (0.20, 0.80),
    "nebius/Qwen/Qwen3-235B-Instruct": (0.20, 0.60),
    "nebius/Qwen/Qwen3-30B-Thinking": (0.10, 0.30),
    "nebius/Qwen/Qwen3-30B-Instruct": (0.10, 0.30),
    "nebius/Qwen/Qwen3-32B": (0.10, 0.30),
    "nebius/Qwen/QwQ-32B": (0.15, 0.45),
    # Nebius Token Factory - Llama models
    "nebius/meta-llama/Llama-3.3-70B-Instruct": (0.13, 0.40),
    "nebius/meta-llama/Llama-3.1-8B-Instruct": (0.02, 0.06),
    "nebius/meta-llama/Llama-3.1-405B-Instruct": (1.00, 3.00),
    # Nebius Token Factory - Other models
    "nebius/gpt-oss-120b": (0.15, 0.60),
    "nebius/gpt-oss-20b": (0.05, 0.20),
    "nebius/THUDM/GLM-4.5": (0.60, 2.20),
    "nebius/Kimi-K2-Instruct": (0.50, 2.40),
    "nebius/NousResearch/Hermes-4-405B": (1.00, 3.00),
    # Local models - free
    "ollama/*": (0.00, 0.00),
    "vllm/*": (0.00, 0.00),
    "local/*": (0.00, 0.00),
}


def calculate_cost(
    model_id: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Calculate cost for API call.

    Args:
        model_id: Model identifier.
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.

    Returns:
        Estimated cost in USD.
    """
    # Check exact match first
    if model_id in MODEL_PRICING:
        input_price, output_price = MODEL_PRICING[model_id]
    # Check wildcard patterns
    elif any(model_id.startswith(p.replace("/*", "")) for p in MODEL_PRICING if "/*" in p):
        return 0.0  # Local models are free
    else:
        # Unknown model, estimate based on similar models
        return 0.0

    input_cost = (input_tokens / 1_000_000) * input_price
    output_cost = (output_tokens / 1_000_000) * output_price
    return input_cost + output_cost


class BaseModelClient(ABC):
    """Abstract base class for model clients.

    Provides common functionality:
    - Retry logic with exponential backoff
    - Cost tracking
    - Statistics collection
    - Rate limiting helpers
    """

    def __init__(
        self,
        model_id: str,
        config: ModelConfig | None = None,
    ) -> None:
        """Initialize model client.

        Args:
            model_id: Model identifier.
            config: Optional configuration.
        """
        self._model_id = model_id
        self._config = config or ModelConfig()
        self._call_count = 0
        self._total_tokens = 0
        self._total_cost = 0.0
        self._total_latency_ms = 0.0

    @property
    def model_id(self) -> str:
        """Return the model identifier."""
        return self._model_id

    @property
    def call_count(self) -> int:
        """Return number of API calls made."""
        return self._call_count

    @property
    def total_tokens(self) -> int:
        """Return total tokens used."""
        return self._total_tokens

    @property
    def total_cost(self) -> float:
        """Return total cost in USD."""
        return self._total_cost

    @property
    def average_latency_ms(self) -> float:
        """Return average latency per call."""
        if self._call_count == 0:
            return 0.0
        return self._total_latency_ms / self._call_count

    def get_statistics(self) -> dict[str, Any]:
        """Return usage statistics.

        Returns:
            Dictionary with call_count, total_tokens, total_cost, average_latency_ms.
        """
        return {
            "model_id": self._model_id,
            "call_count": self._call_count,
            "total_tokens": self._total_tokens,
            "total_cost": self._total_cost,
            "average_latency_ms": self.average_latency_ms,
        }

    def reset_statistics(self) -> None:
        """Reset usage statistics."""
        self._call_count = 0
        self._total_tokens = 0
        self._total_cost = 0.0
        self._total_latency_ms = 0.0

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> ModelResponse:
        """Generate a response from the model.

        Implements retry logic and statistics tracking.

        Args:
            prompt: User prompt.
            system_prompt: Optional system prompt.
            temperature: Sampling temperature (uses default if None).
            max_tokens: Maximum tokens to generate (uses default if None).
            json_mode: Whether to enforce JSON output.

        Returns:
            Model response with text, tokens, and latency.

        Raises:
            RuntimeError: If all retries fail.
        """
        if temperature is None:
            temperature = self._config.default_temperature
        if max_tokens is None:
            max_tokens = self._config.default_max_tokens

        last_error: Exception | None = None
        start_time = time.perf_counter()

        for attempt in range(self._config.max_retries):
            try:
                response = await self._call_api(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=json_mode,
                )

                # Calculate latency
                latency_ms = (time.perf_counter() - start_time) * 1000
                response = ModelResponse(
                    text=response.text,
                    tokens_used=response.tokens_used,
                    latency_ms=latency_ms,
                    model_id=response.model_id,
                    timestamp=response.timestamp,
                    cost_usd=response.cost_usd,
                )

                # Update statistics
                self._call_count += 1
                self._total_tokens += response.tokens_used
                self._total_cost += response.cost_usd
                self._total_latency_ms += latency_ms

                return response

            except Exception as e:
                last_error = e
                if attempt < self._config.max_retries - 1:
                    delay = self._config.retry_delay * (2**attempt)  # Exponential backoff
                    await asyncio.sleep(delay)

        raise RuntimeError(
            f"All {self._config.max_retries} retries failed for {self._model_id}"
        ) from last_error

    @abstractmethod
    async def _call_api(
        self,
        prompt: str,
        system_prompt: str | None,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> ModelResponse:
        """Make the actual API call.

        Subclasses must implement this method.

        Args:
            prompt: User prompt.
            system_prompt: Optional system prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            json_mode: Whether to enforce JSON output.

        Returns:
            Model response (latency will be recalculated by generate()).
        """
        ...

    def _build_messages(
        self,
        prompt: str,
        system_prompt: str | None,
    ) -> list[Mapping[str, str]]:
        """Build message list for chat completion APIs.

        Args:
            prompt: User prompt.
            system_prompt: Optional system prompt.

        Returns:
            List of message dictionaries.
        """
        messages: list[Mapping[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages


class MockModelClient(BaseModelClient):
    """Mock model client for testing.

    Returns configurable responses without making API calls.
    """

    def __init__(
        self,
        model_id: str = "mock-model",
        responses: list[str] | None = None,
        tokens_per_response: int = 100,
        latency_ms: float = 50.0,
    ) -> None:
        """Initialize mock client.

        Args:
            model_id: Model identifier.
            responses: List of responses to return (cycles through).
            tokens_per_response: Tokens to report per response.
            latency_ms: Latency to simulate.
        """
        super().__init__(model_id)
        self._responses = responses or ['{"result": "mock response"}']
        self._response_index = 0
        self._tokens_per_response = tokens_per_response
        self._mock_latency_ms = latency_ms

    async def _call_api(
        self,
        prompt: str,
        system_prompt: str | None,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> ModelResponse:
        """Return mock response.

        Args:
            prompt: User prompt (unused).
            system_prompt: System prompt (unused).
            temperature: Temperature (unused).
            max_tokens: Max tokens (unused).
            json_mode: JSON mode (unused).

        Returns:
            Mock model response.
        """
        response_text = self._responses[self._response_index % len(self._responses)]
        self._response_index += 1

        # Simulate some latency
        await asyncio.sleep(self._mock_latency_ms / 1000)

        return ModelResponse(
            text=response_text,
            tokens_used=self._tokens_per_response,
            latency_ms=self._mock_latency_ms,
            model_id=self._model_id,
            cost_usd=0.0,
        )
