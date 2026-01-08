"""Protocol classes defining interface contracts for UkrQualBench components.

These protocols ensure type-safe interactions between components and enable
dependency injection for testing.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pathlib import Path


@runtime_checkable
class ModelResponse(Protocol):
    """Contract for model responses from API or local models."""

    @property
    def text(self) -> str:
        """The generated text response."""
        ...

    @property
    def tokens_used(self) -> int:
        """Total tokens consumed (input + output)."""
        ...

    @property
    def latency_ms(self) -> float:
        """Response latency in milliseconds."""
        ...

    @property
    def model_id(self) -> str:
        """Identifier of the model that generated this response."""
        ...

    @property
    def timestamp(self) -> datetime:
        """When the response was generated."""
        ...


@runtime_checkable
class JudgeVerdict(Protocol):
    """Contract for judge verdicts in pairwise comparisons."""

    @property
    def winner(self) -> Literal["A", "B", "tie"]:
        """Which response won the comparison."""
        ...

    @property
    def confidence(self) -> Literal["high", "medium", "low"]:
        """Judge's confidence in the verdict."""
        ...

    @property
    def reasoning(self) -> str:
        """Brief explanation of the judgment (1-2 sentences)."""
        ...

    @property
    def raw_response(self) -> str:
        """Raw JSON response from the judge for debugging and audit."""
        ...

    @property
    def latency_ms(self) -> float:
        """Judge response latency in milliseconds."""
        ...


@runtime_checkable
class ComparisonRecord(Protocol):
    """Contract for comparison audit records."""

    @property
    def comparison_id(self) -> str:
        """Idempotent key: hash(prompt_id, model_a, model_b)."""
        ...

    @property
    def prompt_id(self) -> str:
        """ID of the prompt used for comparison."""
        ...

    @property
    def model_a_id(self) -> str:
        """ID of first model in comparison."""
        ...

    @property
    def model_b_id(self) -> str:
        """ID of second model in comparison."""
        ...

    @property
    def response_a(self) -> ModelResponse:
        """Response from model A."""
        ...

    @property
    def response_b(self) -> ModelResponse:
        """Response from model B."""
        ...

    @property
    def verdict(self) -> JudgeVerdict:
        """Judge's verdict on the comparison."""
        ...

    @property
    def judge_id(self) -> str:
        """ID of the judge that made this comparison."""
        ...

    @property
    def timestamp(self) -> datetime:
        """When the comparison was made."""
        ...

    @property
    def position_order(self) -> Literal["AB", "BA"]:
        """Order in which responses were presented to judge (for position bias tracking)."""
        ...


@runtime_checkable
class EvaluationResult(Protocol):
    """Contract for evaluation results."""

    @property
    def model_id(self) -> str:
        """ID of the evaluated model."""
        ...

    @property
    def scores(self) -> ModelScore:
        """Computed scores for all metrics."""
        ...

    @property
    def metadata(self) -> EvaluationMetadata:
        """Evaluation metadata (timing, versions, etc.)."""
        ...

    @property
    def comparisons(self) -> list[ComparisonRecord]:
        """All comparisons this model participated in."""
        ...

    @property
    def checkpoints(self) -> list[str]:
        """Paths to checkpoint files saved during evaluation."""
        ...


@runtime_checkable
class CalibrationResult(Protocol):
    """Contract for judge calibration results."""

    @property
    def judge_id(self) -> str:
        """ID of the calibrated judge."""
        ...

    @property
    def passed(self) -> bool:
        """Whether the judge passed calibration (score >= 0.80)."""
        ...

    @property
    def mc_accuracy(self) -> float:
        """Multiple choice agreement accuracy (threshold: >85%)."""
        ...

    @property
    def gec_f1(self) -> float:
        """Grammar error correction F1 score (threshold: >80%)."""
        ...

    @property
    def russism_f1(self) -> float:
        """Russism detection F1 score (threshold: >85%)."""
        ...

    @property
    def false_positive_rate(self) -> float:
        """False positive rate on correct texts (threshold: <15%)."""
        ...

    @property
    def pairwise_consistency(self) -> float:
        """Consistency with gold standard pairwise judgments (threshold: >90%)."""
        ...

    @property
    def position_bias(self) -> float:
        """Position bias (A-preference rate, threshold: 45-55%)."""
        ...

    @property
    def length_bias_correlation(self) -> float:
        """Correlation with response length (threshold: |r| < 0.3)."""
        ...

    @property
    def final_score(self) -> float:
        """Weighted final calibration score."""
        ...

    @property
    def failure_reasons(self) -> list[str]:
        """List of reasons why calibration failed (if any)."""
        ...

    @property
    def timestamp(self) -> datetime:
        """When calibration was performed."""
        ...


@runtime_checkable
class Checkpoint(Protocol):
    """Contract for checkpoint data for evaluation resumption."""

    @property
    def run_id(self) -> str:
        """Unique identifier for the evaluation run."""
        ...

    @property
    def timestamp(self) -> datetime:
        """When checkpoint was created."""
        ...

    @property
    def benchmark_version(self) -> str:
        """Version of the benchmark (lite/base/large)."""
        ...

    @property
    def completed_comparisons(self) -> int:
        """Number of comparisons completed."""
        ...

    @property
    def total_comparisons(self) -> int:
        """Total number of comparisons planned."""
        ...

    @property
    def pending_pairs(self) -> list[tuple[str, str]]:
        """Model pairs still to be compared."""
        ...

    @property
    def current_elo_ratings(self) -> dict[str, float]:
        """Current ELO ratings for all models."""
        ...

    @property
    def comparison_results(self) -> list[ComparisonRecord]:
        """All comparison results so far."""
        ...

    @property
    def api_costs(self) -> dict[str, float]:
        """Accumulated API costs by provider."""
        ...

    @property
    def errors_logged(self) -> int:
        """Number of errors logged during evaluation."""
        ...

    def save(self, path: Path) -> None:
        """Save checkpoint to disk with atomic write."""
        ...

    @classmethod
    def load(cls, path: Path) -> Checkpoint:
        """Load and validate checkpoint from disk."""
        ...


@runtime_checkable
class ModelScore(Protocol):
    """Contract for final model scores."""

    # Primary ranking metric
    @property
    def elo_rating(self) -> float:
        """Overall ELO rating (1000-2000 range)."""
        ...

    # Block A (calibration tests)
    @property
    def mc_accuracy(self) -> float:
        """Multiple choice accuracy (0-1)."""
        ...

    @property
    def gec_f1(self) -> float:
        """Grammar error correction F1 (0-1)."""
        ...

    @property
    def translation_comet(self) -> float:
        """Translation COMET score (0-1)."""
        ...

    @property
    def false_positive_rate(self) -> float:
        """False positive rate (0-1, lower is better)."""
        ...

    # Block B (generation, via ELO)
    @property
    def generation_elo(self) -> float:
        """ELO for free generation tasks."""
        ...

    @property
    def adversarial_elo(self) -> float:
        """ELO for adversarial tasks."""
        ...

    @property
    def long_context_elo(self) -> float:
        """ELO for long context tasks."""
        ...

    # Block V (automatic metrics)
    @property
    def fertility_rate(self) -> float:
        """Tokens per word ratio (lower is better, target <1.6)."""
        ...

    @property
    def positive_markers(self) -> float:
        """Positive markers per 1000 tokens."""
        ...

    @property
    def russism_rate(self) -> float:
        """Russisms per 1000 tokens (lower is better)."""
        ...

    @property
    def anglicism_rate(self) -> float:
        """Anglicisms per 1000 tokens (lower is better)."""
        ...


@runtime_checkable
class EvaluationMetadata(Protocol):
    """Contract for evaluation metadata."""

    @property
    def benchmark_version(self) -> str:
        """Benchmark version used (lite/base/large)."""
        ...

    @property
    def dataset_hash(self) -> str:
        """SHA-256 hash of the dataset for reproducibility."""
        ...

    @property
    def judge_id(self) -> str:
        """ID of the judge used."""
        ...

    @property
    def judge_calibration_score(self) -> float:
        """Judge's calibration score."""
        ...

    @property
    def total_prompts(self) -> int:
        """Total number of prompts evaluated."""
        ...

    @property
    def total_comparisons(self) -> int:
        """Total number of pairwise comparisons."""
        ...

    @property
    def runtime_minutes(self) -> float:
        """Total evaluation runtime in minutes."""
        ...

    @property
    def total_cost_usd(self) -> float:
        """Total API cost in USD."""
        ...

    @property
    def timestamp(self) -> datetime:
        """When evaluation completed."""
        ...


@runtime_checkable
class EvaluatorHooks(Protocol):
    """Lifecycle hooks for custom extensions and integrations.

    Implement this protocol to add custom behavior at various points
    in the evaluation lifecycle.
    """

    async def on_evaluation_start(
        self, models: list[str], benchmark: str
    ) -> None:
        """Called when evaluation begins.

        Args:
            models: List of model IDs being evaluated.
            benchmark: Benchmark version (lite/base/large).
        """
        ...

    async def on_comparison_start(
        self, model_a: str, model_b: str, prompt_id: str
    ) -> None:
        """Called before each comparison.

        Args:
            model_a: ID of first model.
            model_b: ID of second model.
            prompt_id: ID of the prompt being used.
        """
        ...

    async def on_comparison_complete(self, result: ComparisonRecord) -> None:
        """Called after each comparison (for logging, metrics).

        Args:
            result: The completed comparison record.
        """
        ...

    async def on_round_complete(
        self, round_num: int, rankings: list[tuple[str, float]]
    ) -> None:
        """Called after tournament round completes.

        Args:
            round_num: Which round just completed (1-indexed).
            rankings: Current rankings as (model_id, elo_rating) tuples.
        """
        ...

    async def on_checkpoint_saved(self, checkpoint_path: str) -> None:
        """Called after checkpoint is saved.

        Args:
            checkpoint_path: Path to the saved checkpoint file.
        """
        ...

    async def on_evaluation_complete(self, leaderboard: Leaderboard) -> None:
        """Called when evaluation finishes.

        Args:
            leaderboard: Final leaderboard with all results.
        """
        ...

    async def on_error(self, error: Exception, context: dict[str, str]) -> None:
        """Called when recoverable error occurs.

        Args:
            error: The exception that occurred.
            context: Additional context (prompt_id, model_id, etc.).
        """
        ...


@runtime_checkable
class Leaderboard(Protocol):
    """Contract for the final leaderboard."""

    @property
    def benchmark_version(self) -> str:
        """Benchmark version used."""
        ...

    @property
    def results(self) -> list[EvaluationResult]:
        """Results for all models, sorted by ELO rating."""
        ...

    @property
    def timestamp(self) -> datetime:
        """When leaderboard was generated."""
        ...

    def to_html(self, template_path: Path | None = None) -> str:
        """Render leaderboard as HTML."""
        ...

    def to_json(self) -> str:
        """Export leaderboard as JSON."""
        ...
