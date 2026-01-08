"""Circuit breaker pattern for API resilience.

Implements the circuit breaker pattern to prevent cascading failures
when API providers become unstable. Based on Section 12.2 of the
UkrQualBench Technical Specification.

States:
- CLOSED: Normal operation, requests pass through
- OPEN: Failure threshold exceeded, requests fail fast
- HALF_OPEN: Testing if service recovered, limited requests allowed
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar

T = TypeVar("T")


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open and request is rejected."""

    def __init__(
        self,
        provider: str,
        time_until_half_open: float,
        message: str | None = None,
    ) -> None:
        """Initialize circuit open error.

        Args:
            provider: Name of the API provider.
            time_until_half_open: Seconds until circuit transitions to half-open.
            message: Optional custom message.
        """
        self.provider = provider
        self.time_until_half_open = time_until_half_open
        default_message = (
            f"Circuit breaker for {provider} is OPEN. "
            f"Retry in {time_until_half_open:.1f}s"
        )
        super().__init__(message or default_message)


@dataclass
class CircuitBreaker:
    """Circuit breaker for API call protection.

    Attributes:
        provider: Name of the API provider (for logging/errors).
        failure_threshold: Number of failures before opening circuit.
        success_threshold: Successes needed in half-open to close.
        timeout_seconds: Time before transitioning from open to half-open.
    """

    provider: str
    failure_threshold: int = 5
    success_threshold: int = 3
    timeout_seconds: int = 60
    enabled: bool = True

    # Internal state
    _state: CircuitState = field(default=CircuitState.CLOSED, repr=False)
    _failure_count: int = field(default=0, repr=False)
    _success_count: int = field(default=0, repr=False)
    _last_failure_time: float | None = field(default=None, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    @property
    def state(self) -> CircuitState:
        """Current circuit state."""
        return self._state

    @property
    def failure_count(self) -> int:
        """Current consecutive failure count."""
        return self._failure_count

    @property
    def success_count(self) -> int:
        """Current consecutive success count in half-open state."""
        return self._success_count

    def _should_transition_to_half_open(self) -> bool:
        """Check if enough time passed to try half-open state."""
        if self._state != CircuitState.OPEN:
            return False
        if self._last_failure_time is None:
            return True
        elapsed = time.monotonic() - self._last_failure_time
        return elapsed >= self.timeout_seconds

    def _get_time_until_half_open(self) -> float:
        """Get seconds until circuit transitions to half-open."""
        if self._state != CircuitState.OPEN or self._last_failure_time is None:
            return 0.0
        elapsed = time.monotonic() - self._last_failure_time
        remaining = self.timeout_seconds - elapsed
        return max(0.0, remaining)

    def _record_success(self) -> None:
        """Record a successful call."""
        self._failure_count = 0

        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.success_threshold:
                self._state = CircuitState.CLOSED
                self._success_count = 0

    def _record_failure(self) -> None:
        """Record a failed call."""
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        self._success_count = 0

        if self._state == CircuitState.HALF_OPEN:
            # Any failure in half-open reopens the circuit
            self._state = CircuitState.OPEN
        elif self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN

    async def call(
        self,
        func: Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Execute function through circuit breaker.

        Args:
            func: Async function to execute.
            *args: Positional arguments for func.
            **kwargs: Keyword arguments for func.

        Returns:
            Result from func.

        Raises:
            CircuitOpenError: If circuit is open and request rejected.
            Exception: Any exception from func (after recording failure).
        """
        if not self.enabled:
            return await func(*args, **kwargs)

        async with self._lock:
            # Check if we should transition to half-open
            if self._should_transition_to_half_open():
                self._state = CircuitState.HALF_OPEN
                self._success_count = 0

            # Reject if circuit is open
            if self._state == CircuitState.OPEN:
                raise CircuitOpenError(
                    provider=self.provider,
                    time_until_half_open=self._get_time_until_half_open(),
                )

        # Execute the call
        try:
            result = await func(*args, **kwargs)
            async with self._lock:
                self._record_success()
            return result
        except Exception:
            async with self._lock:
                self._record_failure()
            raise

    def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None

    def force_open(self) -> None:
        """Force circuit to open state (for testing or manual override)."""
        self._state = CircuitState.OPEN
        self._last_failure_time = time.monotonic()

    def force_close(self) -> None:
        """Force circuit to closed state (for manual override)."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0

    def get_status(self) -> dict[str, Any]:
        """Get current circuit breaker status.

        Returns:
            Dict with state, counts, and timing info.
        """
        return {
            "provider": self.provider,
            "state": self._state.value,
            "enabled": self.enabled,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "failure_threshold": self.failure_threshold,
            "success_threshold": self.success_threshold,
            "timeout_seconds": self.timeout_seconds,
            "time_until_half_open": self._get_time_until_half_open(),
        }


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers by provider."""

    def __init__(
        self,
        default_failure_threshold: int = 5,
        default_success_threshold: int = 3,
        default_timeout_seconds: int = 60,
    ) -> None:
        """Initialize registry with default settings.

        Args:
            default_failure_threshold: Default failures before opening.
            default_success_threshold: Default successes to close.
            default_timeout_seconds: Default timeout in seconds.
        """
        self._breakers: dict[str, CircuitBreaker] = {}
        self._default_failure_threshold = default_failure_threshold
        self._default_success_threshold = default_success_threshold
        self._default_timeout_seconds = default_timeout_seconds

    def get(self, provider: str) -> CircuitBreaker:
        """Get or create circuit breaker for provider.

        Args:
            provider: Provider name (e.g., "openai", "anthropic").

        Returns:
            Circuit breaker for the provider.
        """
        if provider not in self._breakers:
            self._breakers[provider] = CircuitBreaker(
                provider=provider,
                failure_threshold=self._default_failure_threshold,
                success_threshold=self._default_success_threshold,
                timeout_seconds=self._default_timeout_seconds,
            )
        return self._breakers[provider]

    def configure(
        self,
        provider: str,
        *,
        failure_threshold: int | None = None,
        success_threshold: int | None = None,
        timeout_seconds: int | None = None,
        enabled: bool | None = None,
    ) -> CircuitBreaker:
        """Configure circuit breaker for a specific provider.

        Args:
            provider: Provider name.
            failure_threshold: Optional override for failure threshold.
            success_threshold: Optional override for success threshold.
            timeout_seconds: Optional override for timeout.
            enabled: Optional enable/disable.

        Returns:
            Configured circuit breaker.
        """
        breaker = self.get(provider)
        if failure_threshold is not None:
            breaker.failure_threshold = failure_threshold
        if success_threshold is not None:
            breaker.success_threshold = success_threshold
        if timeout_seconds is not None:
            breaker.timeout_seconds = timeout_seconds
        if enabled is not None:
            breaker.enabled = enabled
        return breaker

    def get_all_status(self) -> dict[str, dict[str, Any]]:
        """Get status of all circuit breakers.

        Returns:
            Dict mapping provider names to their status.
        """
        return {
            provider: breaker.get_status()
            for provider, breaker in self._breakers.items()
        }

    def reset_all(self) -> None:
        """Reset all circuit breakers to closed state."""
        for breaker in self._breakers.values():
            breaker.reset()
