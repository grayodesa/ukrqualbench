"""Checkpoint system for evaluation state persistence and recovery.

Implements FR-5 (Result Persistence) from the UkrQualBench Technical Specification:
- FR-5.1: Save every 100 comparisons
- FR-5.2: Resume from checkpoint
- FR-5.3: Full audit trail
- FR-5.4: Idempotent keys

Checkpoints enable:
- Recovery from crashes or interruptions
- Resumption of long-running evaluations
- Cost tracking and budget management
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


def generate_comparison_key(
    prompt_id: str, model_a: str, model_b: str
) -> str:
    """Generate deterministic idempotency key for a comparison.

    The key is the same regardless of model order (A,B vs B,A)
    to ensure we don't duplicate comparisons.

    Args:
        prompt_id: ID of the prompt.
        model_a: ID of first model.
        model_b: ID of second model.

    Returns:
        16-character hex string.
    """
    # Sort models to ensure A vs B == B vs A
    models = tuple(sorted([model_a, model_b]))
    data = f"{prompt_id}:{models[0]}:{models[1]}"
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def generate_run_id() -> str:
    """Generate unique run ID for an evaluation session.

    Returns:
        Run ID in format: eval-YYYYMMDD-HHMMSS-XXXX
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    random_suffix = hashlib.sha256(
        str(datetime.now().timestamp()).encode()
    ).hexdigest()[:4]
    return f"eval-{timestamp}-{random_suffix}"


@dataclass
class ComparisonResult:
    """Minimal comparison result for checkpoint storage."""

    comparison_id: str  # Idempotency key
    prompt_id: str
    model_a_id: str
    model_b_id: str
    winner: str  # "A", "B", or "tie"
    confidence: str  # "high", "medium", "low"
    judge_id: str
    timestamp: str  # ISO format
    position_order: str  # "AB" or "BA"
    cost_usd: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ComparisonResult:
        """Create from dictionary."""
        return cls(**data)


@dataclass
class CheckpointData:
    """Complete checkpoint state for an evaluation run.

    This is the data structure that gets serialized to disk.
    """

    run_id: str
    timestamp: str  # ISO format
    benchmark_version: str
    dataset_hash: str
    judge_id: str

    # Progress tracking
    completed_comparisons: int
    total_comparisons: int

    # Pending work
    pending_pairs: list[tuple[str, str]]

    # Results
    current_elo_ratings: dict[str, float]
    comparison_results: list[dict[str, Any]]

    # Cost tracking
    api_costs: dict[str, float] = field(default_factory=dict)
    total_cost_usd: float = 0.0

    # Error tracking
    errors_logged: int = 0
    error_log: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointData:
        """Create from dictionary."""
        # Handle pending_pairs which might be list of lists from JSON
        pending_pairs = [
            tuple(pair) for pair in data.get("pending_pairs", [])
        ]
        data["pending_pairs"] = pending_pairs
        return cls(**data)

    @property
    def progress_percent(self) -> float:
        """Completion percentage."""
        if self.total_comparisons == 0:
            return 0.0
        return (self.completed_comparisons / self.total_comparisons) * 100


class CheckpointManager:
    """Manager for checkpoint save/load operations.

    Handles:
    - Atomic checkpoint saves (write to temp, then rename)
    - Checkpoint validation
    - Finding latest checkpoint for resumption
    - Checkpoint cleanup
    """

    def __init__(self, results_dir: Path) -> None:
        """Initialize checkpoint manager.

        Args:
            results_dir: Base directory for results/checkpoints.
        """
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def get_run_dir(self, run_id: str) -> Path:
        """Get directory for a specific run.

        Args:
            run_id: Evaluation run ID.

        Returns:
            Path to run directory.
        """
        run_dir = self.results_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def save(self, checkpoint: CheckpointData) -> Path:
        """Save checkpoint with atomic write.

        Uses write-to-temp-then-rename pattern to ensure
        we never have a corrupted checkpoint file.

        Args:
            checkpoint: Checkpoint data to save.

        Returns:
            Path to saved checkpoint file.
        """
        run_dir = self.get_run_dir(checkpoint.run_id)
        checkpoint_path = run_dir / "checkpoint.json"
        temp_path = run_dir / "checkpoint.json.tmp"

        # Write to temp file
        data = checkpoint.to_dict()
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Atomic rename
        temp_path.rename(checkpoint_path)

        return checkpoint_path

    def save_round(
        self, checkpoint: CheckpointData, round_num: int
    ) -> Path:
        """Save round-specific checkpoint.

        Args:
            checkpoint: Current checkpoint state.
            round_num: Round number.

        Returns:
            Path to round checkpoint file.
        """
        run_dir = self.get_run_dir(checkpoint.run_id)
        rounds_dir = run_dir / "rounds"
        rounds_dir.mkdir(exist_ok=True)

        round_path = rounds_dir / f"round_{round_num:03d}.json"
        data = checkpoint.to_dict()
        with open(round_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return round_path

    def load(self, run_id: str) -> CheckpointData | None:
        """Load checkpoint for a run.

        Args:
            run_id: Evaluation run ID.

        Returns:
            Checkpoint data if found and valid, None otherwise.
        """
        run_dir = self.results_dir / run_id
        checkpoint_path = run_dir / "checkpoint.json"

        if not checkpoint_path.exists():
            return None

        try:
            with open(checkpoint_path, encoding="utf-8") as f:
                data = json.load(f)
            return CheckpointData.from_dict(data)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # Log error but don't raise - checkpoint is corrupted
            # In production, would log this properly
            print(f"Warning: Corrupted checkpoint at {checkpoint_path}: {e}")
            return None

    def find_latest(self) -> CheckpointData | None:
        """Find most recent checkpoint across all runs.

        Useful for auto-resumption when run_id is not known.

        Returns:
            Most recent valid checkpoint, or None.
        """
        latest: CheckpointData | None = None
        latest_time: datetime | None = None

        for run_dir in self.results_dir.iterdir():
            if not run_dir.is_dir():
                continue

            checkpoint = self.load(run_dir.name)
            if checkpoint is None:
                continue

            try:
                cp_time = datetime.fromisoformat(checkpoint.timestamp)
                if latest_time is None or cp_time > latest_time:
                    latest = checkpoint
                    latest_time = cp_time
            except ValueError:
                continue

        return latest

    def save_final(self, checkpoint: CheckpointData) -> Path:
        """Save final results after evaluation completes.

        Args:
            checkpoint: Final checkpoint state.

        Returns:
            Path to final results file.
        """
        run_dir = self.get_run_dir(checkpoint.run_id)
        final_path = run_dir / "final.json"

        data = checkpoint.to_dict()
        data["completed_at"] = datetime.now().isoformat()

        with open(final_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return final_path

    def cleanup_old(self, keep_count: int = 10) -> list[Path]:
        """Remove old run directories, keeping most recent.

        Args:
            keep_count: Number of recent runs to keep.

        Returns:
            List of removed directory paths.
        """
        runs: list[tuple[datetime, Path]] = []

        for run_dir in self.results_dir.iterdir():
            if not run_dir.is_dir():
                continue

            checkpoint = self.load(run_dir.name)
            if checkpoint is None:
                # No valid checkpoint, check if directory should be kept
                continue

            try:
                cp_time = datetime.fromisoformat(checkpoint.timestamp)
                runs.append((cp_time, run_dir))
            except ValueError:
                continue

        # Sort by time, newest first
        runs.sort(key=lambda x: x[0], reverse=True)

        # Remove old runs
        removed: list[Path] = []
        for _, run_dir in runs[keep_count:]:
            shutil.rmtree(run_dir)
            removed.append(run_dir)

        return removed

    def get_completed_comparisons(
        self, checkpoint: CheckpointData
    ) -> set[str]:
        """Get set of completed comparison IDs for deduplication.

        Args:
            checkpoint: Checkpoint to extract IDs from.

        Returns:
            Set of comparison_id strings.
        """
        return {
            result["comparison_id"]
            for result in checkpoint.comparison_results
        }


def create_initial_checkpoint(
    run_id: str,
    benchmark_version: str,
    dataset_hash: str,
    judge_id: str,
    model_ids: list[str],
    pending_pairs: list[tuple[str, str]],
    initial_elo: float = 1500.0,
) -> CheckpointData:
    """Create initial checkpoint for a new evaluation run.

    Args:
        run_id: Unique run identifier.
        benchmark_version: Benchmark version (lite/base/large).
        dataset_hash: SHA-256 hash of dataset.
        judge_id: ID of the judge being used.
        model_ids: List of model IDs being evaluated.
        pending_pairs: All model pairs to compare.
        initial_elo: Starting ELO rating.

    Returns:
        Initial checkpoint data.
    """
    # Estimate total comparisons (pairs x prompts estimated)
    total_comparisons = len(pending_pairs)

    return CheckpointData(
        run_id=run_id,
        timestamp=datetime.now().isoformat(),
        benchmark_version=benchmark_version,
        dataset_hash=dataset_hash,
        judge_id=judge_id,
        completed_comparisons=0,
        total_comparisons=total_comparisons,
        pending_pairs=list(pending_pairs),
        current_elo_ratings=dict.fromkeys(model_ids, initial_elo),
        comparison_results=[],
        api_costs={},
        total_cost_usd=0.0,
        errors_logged=0,
        error_log=[],
    )
