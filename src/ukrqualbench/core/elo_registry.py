"""Persistent ELO registry for cross-session model ratings.

Implements persistent ELO ratings that accumulate across evaluation runs,
enabling incremental model addition to the leaderboard.

Key features:
- Persistent storage in JSON format
- Atomic writes with backup
- Append-only comparison log for audit trail
- Anchor comparisons for new model calibration
- Statistics tracking per model
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal


@dataclass
class ModelEntry:
    """Persistent record for a model in the registry."""

    model_id: str
    rating: float
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    ties: int = 0
    first_seen: str = ""
    last_updated: str = ""
    is_provisional: bool = True

    def __post_init__(self) -> None:
        """Set timestamps if not provided."""
        now = datetime.now().isoformat()
        if not self.first_seen:
            self.first_seen = now
        if not self.last_updated:
            self.last_updated = now

    @property
    def win_rate(self) -> float:
        """Win rate as proportion (0-1)."""
        if self.games_played == 0:
            return 0.0
        return self.wins / self.games_played

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelEntry:
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ComparisonLogEntry:
    """Single comparison record in the append-only log."""

    timestamp: str
    model_a: str
    model_b: str
    winner: Literal["A", "B", "tie"]
    prompt_id: str | None
    judge_id: str | None
    old_rating_a: float
    old_rating_b: float
    new_rating_a: float
    new_rating_b: float
    run_id: str | None = None
    benchmark_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ComparisonLogEntry:
        """Create from dictionary."""
        return cls(**data)


@dataclass
class RegistryMetadata:
    """Metadata about the registry."""

    created_at: str = ""
    last_updated: str = ""
    total_comparisons: int = 0
    initial_rating: float = 1500.0
    k_factor: float = 32.0
    version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RegistryMetadata:
        """Create from dictionary."""
        return cls(**data)


class ELORegistry:
    """Persistent ELO rating registry.

    Stores model ratings across evaluation sessions, enabling:
    - Incremental model addition
    - Continuous leaderboard updates
    - Full audit trail of comparisons

    Attributes:
        registry_path: Path to the JSON registry file.
        log_path: Path to the comparison log file.
        models: Dictionary of model entries.
        metadata: Registry metadata.
    """

    PROVISIONAL_THRESHOLD = 30

    def __init__(
        self,
        registry_path: Path | str | None = None,
        initial_rating: float = 1500.0,
        k_factor: float = 32.0,
    ) -> None:
        """Initialize ELO registry.

        Args:
            registry_path: Path to registry JSON file. Defaults to data/elo_registry.json.
            initial_rating: Starting rating for new models.
            k_factor: ELO K-factor for rating updates.
        """
        if registry_path is None:
            registry_path = Path("data") / "elo_registry.json"
        self.registry_path = Path(registry_path)
        self.log_path = self.registry_path.with_suffix(".log.json")

        self._initial_rating = initial_rating
        self._k_factor = k_factor

        self.models: dict[str, ModelEntry] = {}
        self.metadata = RegistryMetadata(
            created_at=datetime.now().isoformat(),
            last_updated=datetime.now().isoformat(),
            initial_rating=initial_rating,
            k_factor=k_factor,
        )
        self._comparison_log: list[ComparisonLogEntry] = []
        self._dirty = False
        self._load()

    def _load(self) -> None:
        """Load registry from disk if exists."""
        if not self.registry_path.exists():
            return

        try:
            with open(self.registry_path, encoding="utf-8") as f:
                data = json.load(f)

            if "metadata" in data:
                self.metadata = RegistryMetadata.from_dict(data["metadata"])
                self._initial_rating = self.metadata.initial_rating
                self._k_factor = self.metadata.k_factor

            for model_data in data.get("models", []):
                entry = ModelEntry.from_dict(model_data)
                self.models[entry.model_id] = entry

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            backup_path = self.registry_path.with_suffix(".json.bak")
            shutil.copy2(self.registry_path, backup_path)
            print(f"Warning: Registry corrupted, backed up to {backup_path}: {e}")

        if self.log_path.exists():
            try:
                with open(self.log_path, encoding="utf-8") as f:
                    log_data = json.load(f)
                self._comparison_log = [ComparisonLogEntry.from_dict(entry) for entry in log_data]
            except (json.JSONDecodeError, KeyError, TypeError):
                self._comparison_log = []

    def save(self) -> Path:
        """Save registry to disk with atomic write.

        Returns:
            Path to saved registry file.
        """
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata.last_updated = datetime.now().isoformat()
        self.metadata.total_comparisons = len(self._comparison_log)

        data = {
            "metadata": self.metadata.to_dict(),
            "models": [entry.to_dict() for entry in self.models.values()],
        }

        temp_path = self.registry_path.with_suffix(".json.tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        temp_path.rename(self.registry_path)

        log_temp = self.log_path.with_suffix(".log.json.tmp")
        with open(log_temp, "w", encoding="utf-8") as f:
            json.dump(
                [entry.to_dict() for entry in self._comparison_log],
                f,
                indent=2,
                ensure_ascii=False,
            )
        log_temp.rename(self.log_path)

        self._dirty = False
        return self.registry_path

    def register_model(self, model_id: str, rating: float | None = None) -> ModelEntry:
        """Register a new model or return existing.

        Args:
            model_id: Unique model identifier.
            rating: Optional starting rating (defaults to initial_rating).

        Returns:
            ModelEntry for the model.
        """
        if model_id in self.models:
            return self.models[model_id]

        entry = ModelEntry(
            model_id=model_id,
            rating=rating if rating is not None else self._initial_rating,
            is_provisional=True,
        )
        self.models[model_id] = entry
        self._dirty = True
        return entry

    def get_rating(self, model_id: str) -> float:
        """Get current rating for a model.

        Args:
            model_id: Model identifier.

        Returns:
            Current ELO rating.

        Raises:
            KeyError: If model not registered.
        """
        if model_id not in self.models:
            raise KeyError(f"Model {model_id} not in registry")
        return self.models[model_id].rating

    def get_model(self, model_id: str) -> ModelEntry | None:
        """Get model entry if exists.

        Args:
            model_id: Model identifier.

        Returns:
            ModelEntry or None if not found.
        """
        return self.models.get(model_id)

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score for player A against player B.

        Args:
            rating_a: Rating of player A.
            rating_b: Rating of player B.

        Returns:
            Expected score for player A (between 0 and 1).
        """
        exponent = (rating_b - rating_a) / 400.0
        return 1.0 / (1.0 + math.pow(10, exponent))

    def record_comparison(
        self,
        model_a: str,
        model_b: str,
        winner: Literal["A", "B", "tie"],
        *,
        prompt_id: str | None = None,
        judge_id: str | None = None,
        run_id: str | None = None,
        benchmark_version: str | None = None,
    ) -> ComparisonLogEntry:
        """Record a comparison result and update ratings.

        Args:
            model_a: ID of first model.
            model_b: ID of second model.
            winner: "A" if model_a won, "B" if model_b won, "tie" for draw.
            prompt_id: Optional prompt ID for tracking.
            judge_id: Optional judge model ID.
            run_id: Optional evaluation run ID.
            benchmark_version: Optional benchmark version.

        Returns:
            ComparisonLogEntry with details.
        """
        entry_a = self.register_model(model_a)
        entry_b = self.register_model(model_b)

        old_rating_a = entry_a.rating
        old_rating_b = entry_b.rating
        expected_a = self.expected_score(old_rating_a, old_rating_b)
        expected_b = 1.0 - expected_a

        if winner == "A":
            actual_a, actual_b = 1.0, 0.0
            entry_a.wins += 1
            entry_b.losses += 1
        elif winner == "B":
            actual_a, actual_b = 0.0, 1.0
            entry_a.losses += 1
            entry_b.wins += 1
        else:
            actual_a, actual_b = 0.5, 0.5
            entry_a.ties += 1
            entry_b.ties += 1

        new_rating_a = old_rating_a + self._k_factor * (actual_a - expected_a)
        new_rating_b = old_rating_b + self._k_factor * (actual_b - expected_b)

        now = datetime.now().isoformat()
        entry_a.rating = new_rating_a
        entry_a.games_played += 1
        entry_a.last_updated = now
        entry_a.is_provisional = entry_a.games_played < self.PROVISIONAL_THRESHOLD

        entry_b.rating = new_rating_b
        entry_b.games_played += 1
        entry_b.last_updated = now
        entry_b.is_provisional = entry_b.games_played < self.PROVISIONAL_THRESHOLD

        log_entry = ComparisonLogEntry(
            timestamp=now,
            model_a=model_a,
            model_b=model_b,
            winner=winner,
            prompt_id=prompt_id,
            judge_id=judge_id,
            old_rating_a=old_rating_a,
            old_rating_b=old_rating_b,
            new_rating_a=new_rating_a,
            new_rating_b=new_rating_b,
            run_id=run_id,
            benchmark_version=benchmark_version,
        )
        self._comparison_log.append(log_entry)
        self._dirty = True

        return log_entry

    def get_rankings(self) -> list[tuple[str, float]]:
        """Get models ranked by ELO rating (highest first).

        Returns:
            List of (model_id, rating) tuples sorted by rating.
        """
        return sorted(
            [(m.model_id, m.rating) for m in self.models.values()],
            key=lambda x: x[1],
            reverse=True,
        )

    def get_leaderboard(self) -> list[dict[str, Any]]:
        """Get formatted leaderboard with full statistics.

        Returns:
            List of dicts with rank, model_id, rating, games, wins, etc.
        """
        rankings = self.get_rankings()
        leaderboard = []
        for rank, (model_id, rating) in enumerate(rankings, start=1):
            entry = self.models[model_id]
            leaderboard.append(
                {
                    "rank": rank,
                    "model_id": model_id,
                    "rating": round(rating, 1),
                    "games": entry.games_played,
                    "wins": entry.wins,
                    "losses": entry.losses,
                    "ties": entry.ties,
                    "win_rate": round(entry.win_rate, 3),
                    "provisional": entry.is_provisional,
                    "first_seen": entry.first_seen,
                    "last_updated": entry.last_updated,
                }
            )
        return leaderboard

    def get_anchor_models(self, count: int = 3) -> list[str]:
        """Get top-rated non-provisional models for anchoring new models.

        When adding a new model, it should be compared against anchor models
        to calibrate its rating relative to the existing leaderboard.

        Args:
            count: Number of anchor models to return.

        Returns:
            List of model IDs to use as anchors.
        """
        candidates = [
            (m.model_id, m.rating, m.games_played)
            for m in self.models.values()
            if not m.is_provisional
        ]

        if not candidates:
            candidates = [
                (m.model_id, m.rating, m.games_played)
                for m in self.models.values()
                if m.games_played > 0
            ]

        if not candidates:
            return []

        candidates.sort(key=lambda x: x[1], reverse=True)

        if len(candidates) <= count:
            return [c[0] for c in candidates]

        indices = [0]
        if count >= 2:
            indices.append(len(candidates) - 1)
        if count >= 3:
            indices.append(len(candidates) // 2)

        while len(indices) < count and len(indices) < len(candidates):
            sorted_indices = sorted(indices)
            max_gap = 0
            gap_start = 0
            for i in range(len(sorted_indices) - 1):
                gap = sorted_indices[i + 1] - sorted_indices[i]
                if gap > max_gap:
                    max_gap = gap
                    gap_start = sorted_indices[i]
            new_idx = gap_start + max_gap // 2
            if new_idx not in indices:
                indices.append(new_idx)
            else:
                break

        return [candidates[i][0] for i in sorted(indices)[:count]]

    def get_new_models(self, model_ids: list[str]) -> list[str]:
        """Identify which models are new (not in registry).

        Args:
            model_ids: List of model IDs to check.

        Returns:
            List of model IDs not yet in registry.
        """
        return [m for m in model_ids if m not in self.models]

    def get_existing_models(self, model_ids: list[str]) -> list[str]:
        """Identify which models already exist in registry.

        Args:
            model_ids: List of model IDs to check.

        Returns:
            List of model IDs already in registry.
        """
        return [m for m in model_ids if m in self.models]

    def get_comparison_count(self, model_a: str, model_b: str) -> int:
        """Get number of comparisons between two models.

        Args:
            model_a: First model ID.
            model_b: Second model ID.

        Returns:
            Number of recorded comparisons.
        """
        count = 0
        for entry in self._comparison_log:
            if (entry.model_a == model_a and entry.model_b == model_b) or (
                entry.model_a == model_b and entry.model_b == model_a
            ):
                count += 1
        return count

    def get_head_to_head(self, model_a: str, model_b: str) -> dict[str, int]:
        """Get head-to-head record between two models.

        Args:
            model_a: First model ID.
            model_b: Second model ID.

        Returns:
            Dict with 'a_wins', 'b_wins', 'ties'.
        """
        a_wins = 0
        b_wins = 0
        ties = 0

        for entry in self._comparison_log:
            if entry.model_a == model_a and entry.model_b == model_b:
                if entry.winner == "A":
                    a_wins += 1
                elif entry.winner == "B":
                    b_wins += 1
                else:
                    ties += 1
            elif entry.model_a == model_b and entry.model_b == model_a:
                if entry.winner == "A":
                    b_wins += 1
                elif entry.winner == "B":
                    a_wins += 1
                else:
                    ties += 1

        return {"a_wins": a_wins, "b_wins": b_wins, "ties": ties}

    def get_recent_comparisons(self, limit: int = 50) -> list[ComparisonLogEntry]:
        """Get most recent comparisons.

        Args:
            limit: Maximum number to return.

        Returns:
            List of recent ComparisonLogEntry objects.
        """
        return self._comparison_log[-limit:]

    def get_model_history(self, model_id: str) -> list[ComparisonLogEntry]:
        """Get all comparisons involving a model.

        Args:
            model_id: Model identifier.

        Returns:
            List of ComparisonLogEntry objects.
        """
        return [
            entry
            for entry in self._comparison_log
            if entry.model_a == model_id or entry.model_b == model_id
        ]

    def reset(self) -> None:
        """Reset registry to empty state (does not delete files)."""
        self.models.clear()
        self._comparison_log.clear()
        self.metadata = RegistryMetadata(
            created_at=datetime.now().isoformat(),
            last_updated=datetime.now().isoformat(),
            initial_rating=self._initial_rating,
            k_factor=self._k_factor,
        )
        self._dirty = True

    def delete_files(self) -> None:
        """Delete registry files from disk."""
        if self.registry_path.exists():
            self.registry_path.unlink()
        if self.log_path.exists():
            self.log_path.unlink()

    @property
    def model_count(self) -> int:
        """Number of registered models."""
        return len(self.models)

    @property
    def comparison_count(self) -> int:
        """Total number of recorded comparisons."""
        return len(self._comparison_log)

    @property
    def is_dirty(self) -> bool:
        """Whether there are unsaved changes."""
        return self._dirty

    def __contains__(self, model_id: str) -> bool:
        """Check if model is in registry."""
        return model_id in self.models

    def __len__(self) -> int:
        """Number of registered models."""
        return len(self.models)


def create_registry(
    path: Path | str | None = None,
    initial_rating: float = 1500.0,
    k_factor: float = 32.0,
) -> ELORegistry:
    """Factory function to create an ELO registry.

    Args:
        path: Path to registry file. Defaults to data/elo_registry.json.
        initial_rating: Starting ELO rating for new models.
        k_factor: ELO K-factor for rating changes.

    Returns:
        Configured ELORegistry instance.
    """
    return ELORegistry(
        registry_path=path,
        initial_rating=initial_rating,
        k_factor=k_factor,
    )
