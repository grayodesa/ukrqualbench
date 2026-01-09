"""Tests for circuit breaker pattern implementation."""

from __future__ import annotations

import pytest

from ukrqualbench.core.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerRegistry,
    CircuitOpenError,
    CircuitState,
)


class TestCircuitBreaker:
    """Tests for CircuitBreaker class."""

    @pytest.fixture
    def breaker(self) -> CircuitBreaker:
        """Create test circuit breaker with low thresholds."""
        return CircuitBreaker(
            provider="test",
            failure_threshold=3,
            success_threshold=2,
            timeout_seconds=1,
        )

    def test_initial_state_closed(self, breaker: CircuitBreaker) -> None:
        """Test that circuit starts in closed state."""
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0

    @pytest.mark.asyncio
    async def test_success_keeps_closed(self, breaker: CircuitBreaker) -> None:
        """Test successful calls keep circuit closed."""

        async def success() -> str:
            return "ok"

        result = await breaker.call(success)
        assert result == "ok"
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0

    @pytest.mark.asyncio
    async def test_failures_open_circuit(self, breaker: CircuitBreaker) -> None:
        """Test that enough failures open the circuit."""

        async def failure() -> None:
            raise ValueError("fail")

        # Fail 3 times (threshold)
        for _ in range(3):
            with pytest.raises(ValueError):
                await breaker.call(failure)

        assert breaker.state == CircuitState.OPEN
        assert breaker.failure_count == 3

    @pytest.mark.asyncio
    async def test_open_circuit_rejects(self, breaker: CircuitBreaker) -> None:
        """Test that open circuit rejects calls."""
        # Force open
        breaker.force_open()

        async def success() -> str:
            return "ok"

        with pytest.raises(CircuitOpenError) as exc_info:
            await breaker.call(success)

        assert exc_info.value.provider == "test"

    @pytest.mark.asyncio
    async def test_half_open_transition(self, breaker: CircuitBreaker) -> None:
        """Test transition to half-open after timeout."""
        breaker.force_open()
        breaker.timeout_seconds = 0  # Immediate transition for test

        async def success() -> str:
            return "ok"

        # Should transition to half-open and allow call
        result = await breaker.call(success)
        assert result == "ok"
        # After one success in half-open, still half-open (need 2)
        assert breaker.state == CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_half_open_success_closes(self, breaker: CircuitBreaker) -> None:
        """Test that successes in half-open close circuit."""
        breaker.force_open()
        breaker.timeout_seconds = 0
        breaker.success_threshold = 2

        async def success() -> str:
            return "ok"

        # First success - still half-open
        await breaker.call(success)
        assert breaker.state == CircuitState.HALF_OPEN

        # Second success - should close
        await breaker.call(success)
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_half_open_failure_reopens(self, breaker: CircuitBreaker) -> None:
        """Test that failure in half-open reopens circuit."""
        breaker.force_open()
        breaker.timeout_seconds = 0

        async def failure() -> None:
            raise ValueError("fail")

        # Trigger half-open with a success first
        async def success() -> str:
            return "ok"

        await breaker.call(success)
        assert breaker.state == CircuitState.HALF_OPEN

        # Now fail - should reopen
        with pytest.raises(ValueError):
            await breaker.call(failure)

        assert breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_disabled_breaker_passthrough(self, breaker: CircuitBreaker) -> None:
        """Test that disabled breaker passes through."""
        breaker.enabled = False
        breaker.force_open()  # Even when forced open

        async def success() -> str:
            return "ok"

        # Should still work
        result = await breaker.call(success)
        assert result == "ok"

    def test_reset(self, breaker: CircuitBreaker) -> None:
        """Test reset restores initial state."""
        breaker.force_open()
        breaker.reset()

        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0
        assert breaker.success_count == 0

    def test_get_status(self, breaker: CircuitBreaker) -> None:
        """Test status reporting."""
        status = breaker.get_status()

        assert status["provider"] == "test"
        assert status["state"] == "closed"
        assert status["enabled"] is True
        assert status["failure_threshold"] == 3


class TestCircuitBreakerRegistry:
    """Tests for CircuitBreakerRegistry."""

    def test_get_creates_breaker(self) -> None:
        """Test get creates new breaker if not exists."""
        registry = CircuitBreakerRegistry()
        breaker = registry.get("openai")

        assert breaker.provider == "openai"
        assert breaker.state == CircuitState.CLOSED

    def test_get_returns_same_breaker(self) -> None:
        """Test get returns same breaker for same provider."""
        registry = CircuitBreakerRegistry()
        breaker1 = registry.get("openai")
        breaker2 = registry.get("openai")

        assert breaker1 is breaker2

    def test_configure(self) -> None:
        """Test configure updates breaker settings."""
        registry = CircuitBreakerRegistry()
        breaker = registry.configure(
            "anthropic",
            failure_threshold=10,
            timeout_seconds=120,
        )

        assert breaker.failure_threshold == 10
        assert breaker.timeout_seconds == 120

    def test_get_all_status(self) -> None:
        """Test getting status of all breakers."""
        registry = CircuitBreakerRegistry()
        registry.get("openai")
        registry.get("anthropic")

        status = registry.get_all_status()

        assert "openai" in status
        assert "anthropic" in status

    def test_reset_all(self) -> None:
        """Test reset_all resets all breakers."""
        registry = CircuitBreakerRegistry()
        openai = registry.get("openai")
        anthropic = registry.get("anthropic")

        openai.force_open()
        anthropic.force_open()

        registry.reset_all()

        assert openai.state == CircuitState.CLOSED
        assert anthropic.state == CircuitState.CLOSED
