"""Metrics collection for observability.

Implements Section 13 (Observability) from the Technical Specification:
- Counter metrics for comparisons, API calls, calibrations
- Histogram metrics for latencies
- Gauge metrics for active evaluations, budget, etc.

Compatible with OpenTelemetry and Prometheus export patterns.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class Histogram:
    """Simple histogram for latency tracking.

    Buckets follow Prometheus-style exponential distribution.
    """

    name: str
    buckets: list[float] = field(
        default_factory=lambda: [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
    )
    _values: list[float] = field(default_factory=list, repr=False)

    def observe(self, value: float) -> None:
        """Record a value."""
        self._values.append(value)

    @property
    def count(self) -> int:
        """Total number of observations."""
        return len(self._values)

    @property
    def sum(self) -> float:
        """Sum of all values."""
        return sum(self._values)

    def percentile(self, p: float) -> float:
        """Get percentile value.

        Args:
            p: Percentile (0-100).

        Returns:
            Value at percentile, or 0 if no data.
        """
        if not self._values:
            return 0.0
        sorted_values = sorted(self._values)
        idx = int(len(sorted_values) * (p / 100))
        idx = min(idx, len(sorted_values) - 1)
        return sorted_values[idx]

    def bucket_counts(self) -> dict[float, int]:
        """Get count of values in each bucket.

        Returns:
            Dict mapping bucket upper bound to count.
        """
        counts: dict[float, int] = dict.fromkeys(self.buckets, 0)
        counts[float("inf")] = 0

        for value in self._values:
            for bucket in self.buckets:
                if value <= bucket:
                    counts[bucket] += 1
                    break
            else:
                counts[float("inf")] += 1

        return counts


@dataclass
class MetricsCollector:
    """Central metrics collection for UkrQualBench.

    Collects:
    - Comparison counts and durations
    - API request counts, latencies, and costs
    - Detector performance metrics
    - Calibration results
    - Resource utilization
    """

    # Counters
    comparisons_total: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    api_requests_total: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    api_tokens_total: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    calibrations_total: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    comparisons_skipped: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    errors_total: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Histograms
    comparison_duration: Histogram = field(
        default_factory=lambda: Histogram("comparison_duration_seconds")
    )
    api_latency: dict[str, Histogram] = field(default_factory=dict)
    detector_latency: dict[str, Histogram] = field(default_factory=dict)

    # Gauges (current values)
    active_evaluations: int = 0
    current_round: dict[str, int] = field(default_factory=dict)
    budget_remaining_usd: dict[str, float] = field(default_factory=dict)
    elo_ratings: dict[str, dict[str, float]] = field(default_factory=dict)
    circuit_breaker_state: dict[str, int] = field(default_factory=dict)

    # Cost tracking
    costs_by_provider: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    total_cost_usd: float = 0.0

    # Timestamps
    _start_time: float = field(default_factory=time.time, repr=False)
    _last_update: float = field(default_factory=time.time, repr=False)

    def record_comparison(
        self,
        judge: str,
        status: str,
        duration_seconds: float,
    ) -> None:
        """Record a comparison result.

        Args:
            judge: Judge model ID.
            status: "success", "error", or "timeout".
            duration_seconds: Comparison duration.
        """
        key = f"{judge}:{status}"
        self.comparisons_total[key] += 1
        self.comparison_duration.observe(duration_seconds)
        self._last_update = time.time()

    def record_api_request(
        self,
        provider: str,
        status: str,
        latency_seconds: float,
        cost_usd: float,
        tokens_in: int = 0,
        tokens_out: int = 0,
    ) -> None:
        """Record an API request.

        Args:
            provider: API provider (openai, anthropic, google).
            status: "success" or "error".
            latency_seconds: Request latency.
            cost_usd: Cost of this request.
            tokens_in: Input tokens.
            tokens_out: Output tokens.
        """
        key = f"{provider}:{status}"
        self.api_requests_total[key] += 1

        # Tokens
        self.api_tokens_total[f"{provider}:input"] += tokens_in
        self.api_tokens_total[f"{provider}:output"] += tokens_out

        # Latency histogram
        if provider not in self.api_latency:
            self.api_latency[provider] = Histogram(f"api_latency_{provider}")
        self.api_latency[provider].observe(latency_seconds)

        # Cost
        self.costs_by_provider[provider] += cost_usd
        self.total_cost_usd += cost_usd

        self._last_update = time.time()

    def record_detector(
        self,
        detector: str,
        latency_seconds: float,
        tokens_processed: int,
    ) -> None:
        """Record detector performance.

        Args:
            detector: Detector name (russism, anglicism, markers, fertility).
            latency_seconds: Processing time.
            tokens_processed: Number of tokens processed.
        """
        if detector not in self.detector_latency:
            self.detector_latency[detector] = Histogram(f"detector_latency_{detector}")

        # Normalize to per-1K tokens
        if tokens_processed > 0:
            normalized = (latency_seconds / tokens_processed) * 1000
            self.detector_latency[detector].observe(normalized)

        self._last_update = time.time()

    def record_calibration(
        self,
        judge: str,
        passed: bool,
    ) -> None:
        """Record calibration result.

        Args:
            judge: Judge model ID.
            passed: Whether calibration passed.
        """
        result = "pass" if passed else "fail"
        key = f"{judge}:{result}"
        self.calibrations_total[key] += 1
        self._last_update = time.time()

    def record_skip(self, reason: str) -> None:
        """Record a skipped comparison.

        Args:
            reason: Why comparison was skipped (rate_limit, error, budget).
        """
        self.comparisons_skipped[reason] += 1
        self._last_update = time.time()

    def record_error(self, error_type: str, provider: str | None = None) -> None:
        """Record an error.

        Args:
            error_type: Type of error.
            provider: Optional provider name.
        """
        key = f"{provider}:{error_type}" if provider else error_type
        self.errors_total[key] += 1
        self._last_update = time.time()

    def set_circuit_breaker_state(self, provider: str, state: int) -> None:
        """Set circuit breaker state.

        Args:
            provider: Provider name.
            state: 0=closed, 1=open, 2=half_open.
        """
        self.circuit_breaker_state[provider] = state

    def set_budget_remaining(self, evaluation_id: str, amount_usd: float) -> None:
        """Set remaining budget for an evaluation.

        Args:
            evaluation_id: Evaluation run ID.
            amount_usd: Remaining budget in USD.
        """
        self.budget_remaining_usd[evaluation_id] = amount_usd

    def set_current_round(self, evaluation_id: str, round_num: int) -> None:
        """Set current round for an evaluation.

        Args:
            evaluation_id: Evaluation run ID.
            round_num: Current round number.
        """
        self.current_round[evaluation_id] = round_num

    def update_elo_ratings(self, evaluation_id: str, ratings: dict[str, float]) -> None:
        """Update ELO ratings for an evaluation.

        Args:
            evaluation_id: Evaluation run ID.
            ratings: Dict of model_id -> rating.
        """
        self.elo_ratings[evaluation_id] = ratings.copy()

    def get_summary(self) -> dict[str, Any]:
        """Get metrics summary.

        Returns:
            Dict with all current metrics.
        """
        uptime = time.time() - self._start_time

        return {
            "uptime_seconds": uptime,
            "last_update": datetime.fromtimestamp(self._last_update).isoformat(),
            "counters": {
                "comparisons_total": dict(self.comparisons_total),
                "api_requests_total": dict(self.api_requests_total),
                "api_tokens_total": dict(self.api_tokens_total),
                "calibrations_total": dict(self.calibrations_total),
                "comparisons_skipped": dict(self.comparisons_skipped),
                "errors_total": dict(self.errors_total),
            },
            "histograms": {
                "comparison_duration": {
                    "count": self.comparison_duration.count,
                    "p50": self.comparison_duration.percentile(50),
                    "p95": self.comparison_duration.percentile(95),
                    "p99": self.comparison_duration.percentile(99),
                },
                "api_latency": {
                    provider: {
                        "count": hist.count,
                        "p50": hist.percentile(50),
                        "p95": hist.percentile(95),
                        "p99": hist.percentile(99),
                    }
                    for provider, hist in self.api_latency.items()
                },
                "detector_latency": {
                    detector: {
                        "count": hist.count,
                        "p50_ms_per_1k": hist.percentile(50) * 1000,
                        "p95_ms_per_1k": hist.percentile(95) * 1000,
                    }
                    for detector, hist in self.detector_latency.items()
                },
            },
            "gauges": {
                "active_evaluations": self.active_evaluations,
                "current_round": dict(self.current_round),
                "budget_remaining_usd": dict(self.budget_remaining_usd),
                "circuit_breaker_state": dict(self.circuit_breaker_state),
            },
            "costs": {
                "by_provider": dict(self.costs_by_provider),
                "total_usd": self.total_cost_usd,
            },
        }

    def to_prometheus_format(self) -> str:
        """Export metrics in Prometheus text format.

        Returns:
            Prometheus-compatible metrics text.
        """
        lines: list[str] = []

        # Counters
        lines.append("# HELP ukrqualbench_comparisons_total Total comparisons")
        lines.append("# TYPE ukrqualbench_comparisons_total counter")
        for key, count in self.comparisons_total.items():
            judge, status = key.split(":", 1)
            lines.append(
                f'ukrqualbench_comparisons_total{{judge="{judge}",status="{status}"}} {count}'
            )

        lines.append("")
        lines.append("# HELP ukrqualbench_api_requests_total Total API requests")
        lines.append("# TYPE ukrqualbench_api_requests_total counter")
        for key, count in self.api_requests_total.items():
            provider, status = key.split(":", 1)
            lines.append(
                f'ukrqualbench_api_requests_total{{provider="{provider}",status="{status}"}} {count}'
            )

        # Gauges
        lines.append("")
        lines.append("# HELP ukrqualbench_active_evaluations Active evaluations")
        lines.append("# TYPE ukrqualbench_active_evaluations gauge")
        lines.append(f"ukrqualbench_active_evaluations {self.active_evaluations}")

        lines.append("")
        lines.append("# HELP ukrqualbench_total_cost_usd Total cost in USD")
        lines.append("# TYPE ukrqualbench_total_cost_usd gauge")
        lines.append(f"ukrqualbench_total_cost_usd {self.total_cost_usd:.4f}")

        # Circuit breaker state
        lines.append("")
        lines.append("# HELP ukrqualbench_circuit_breaker_state Circuit breaker state")
        lines.append("# TYPE ukrqualbench_circuit_breaker_state gauge")
        for provider, state in self.circuit_breaker_state.items():
            lines.append(f'ukrqualbench_circuit_breaker_state{{provider="{provider}"}} {state}')

        return "\n".join(lines)

    def reset(self) -> None:
        """Reset all metrics to initial state."""
        self.comparisons_total.clear()
        self.api_requests_total.clear()
        self.api_tokens_total.clear()
        self.calibrations_total.clear()
        self.comparisons_skipped.clear()
        self.errors_total.clear()
        self.comparison_duration = Histogram("comparison_duration_seconds")
        self.api_latency.clear()
        self.detector_latency.clear()
        self.active_evaluations = 0
        self.current_round.clear()
        self.budget_remaining_usd.clear()
        self.elo_ratings.clear()
        self.circuit_breaker_state.clear()
        self.costs_by_provider.clear()
        self.total_cost_usd = 0.0
        self._start_time = time.time()
        self._last_update = time.time()


# Global metrics instance for easy access
_global_metrics: MetricsCollector | None = None


def get_metrics() -> MetricsCollector:
    """Get global metrics collector instance.

    Returns:
        Global MetricsCollector singleton.
    """
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = MetricsCollector()
    return _global_metrics


def reset_metrics() -> None:
    """Reset global metrics to initial state."""
    global _global_metrics
    _global_metrics = MetricsCollector()
