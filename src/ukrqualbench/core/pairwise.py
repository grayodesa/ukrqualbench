"""Pairwise comparison engine for model evaluation.

Implements Swiss-system tournament logic for efficient model comparison:
- Smart pairing based on current ELO ratings
- Position randomization to avoid bias
- Comparison scheduling and tracking
- Tie detection and handling
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

from ukrqualbench.core.elo import ELOCalculator
from ukrqualbench.core.elo_registry import ELORegistry
from ukrqualbench.core.schemas import (
    ComparisonRecordData,
    JudgeVerdictData,
    ModelResponseData,
    PositionOrder,
    WinnerChoice,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ukrqualbench.judges.base import BaseJudge
    from ukrqualbench.models.base import BaseModelClient


class PairingStrategy(str, Enum):
    """Strategy for pairing models in tournament."""

    SWISS = "swiss"  # Pair models with similar ratings
    ROUND_ROBIN = "round_robin"  # All pairs (expensive)
    RANDOM = "random"  # Random pairing


@dataclass
class ScheduledComparison:
    """A scheduled comparison between two models."""

    model_a: str
    model_b: str
    prompt_id: str
    position_order: PositionOrder  # Which model is shown first to judge
    comparison_id: str = ""

    def __post_init__(self) -> None:
        """Generate comparison ID if not provided."""
        if not self.comparison_id:
            self.comparison_id = self._generate_id()

    def _generate_id(self) -> str:
        """Generate deterministic comparison ID."""
        # Sort model IDs to ensure same ID regardless of order
        models = sorted([self.model_a, self.model_b])
        key = f"{self.prompt_id}:{models[0]}:{models[1]}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]


@dataclass
class ComparisonResult:
    """Result of a single comparison."""

    scheduled: ScheduledComparison
    response_a: ModelResponseData
    response_b: ModelResponseData
    verdict: JudgeVerdictData
    judge_id: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_record(self) -> ComparisonRecordData:
        """Convert to ComparisonRecordData for storage."""
        return ComparisonRecordData(
            comparison_id=self.scheduled.comparison_id,
            prompt_id=self.scheduled.prompt_id,
            model_a_id=self.scheduled.model_a,
            model_b_id=self.scheduled.model_b,
            response_a=self.response_a,
            response_b=self.response_b,
            verdict=self.verdict,
            judge_id=self.judge_id,
            timestamp=self.timestamp,
            position_order=self.scheduled.position_order,
        )

    @property
    def winner_model_id(self) -> str | None:
        """Get the winning model ID, or None for tie."""
        if self.verdict.winner == WinnerChoice.A:
            return self.scheduled.model_a
        elif self.verdict.winner == WinnerChoice.B:
            return self.scheduled.model_b
        return None


@dataclass
class TournamentRound:
    """A single round of the tournament."""

    round_number: int
    comparisons: list[ScheduledComparison] = field(default_factory=list)
    results: list[ComparisonResult] = field(default_factory=list)
    completed: bool = False

    @property
    def progress(self) -> float:
        """Completion progress (0-1)."""
        if not self.comparisons:
            return 1.0
        return len(self.results) / len(self.comparisons)


class PairwiseEngine:
    """Engine for managing pairwise comparisons in a tournament.

    Implements Swiss-system tournament pairing for efficient evaluation:
    - Models with similar ratings are paired together
    - Position (A/B) is randomized to avoid position bias
    - Comparisons are tracked to avoid duplicates

    Attributes:
        elo_calculator: ELO rating calculator.
        strategy: Pairing strategy (swiss, round_robin, random).
        completed_comparisons: Set of completed comparison IDs.
        comparison_history: List of all comparison results.
    """

    def __init__(
        self,
        initial_rating: float = 1500.0,
        k_factor: float = 32.0,
        strategy: PairingStrategy = PairingStrategy.SWISS,
        seed: int | None = None,
        registry: ELORegistry | None = None,
    ) -> None:
        """Initialize pairwise engine.

        Args:
            initial_rating: Starting ELO rating for models.
            k_factor: ELO K-factor for rating updates.
            strategy: Pairing strategy for tournament.
            seed: Random seed for reproducibility.
            registry: Optional persistent ELO registry for cross-session ratings.
        """
        self.elo_calculator = ELOCalculator(
            initial_rating=initial_rating,
            k_factor=k_factor,
        )
        self.strategy = strategy
        self._rng = random.Random(seed)
        self._completed_comparison_ids: set[str] = set()
        self._comparison_history: list[ComparisonResult] = []
        self._rounds: list[TournamentRound] = []
        self._model_pair_counts: dict[tuple[str, str], int] = {}
        self._registry = registry
        self._run_id: str | None = None
        self._benchmark_version: str | None = None

    @property
    def ratings(self) -> dict[str, float]:
        """Current ELO ratings for all models."""
        if self._registry:
            return {m: e.rating for m, e in self._registry.models.items()}
        return dict(self.elo_calculator.ratings)

    @property
    def registry(self) -> ELORegistry | None:
        """Persistent ELO registry if configured."""
        return self._registry

    def set_run_metadata(self, run_id: str, benchmark_version: str) -> None:
        """Set metadata for comparison logging."""
        self._run_id = run_id
        self._benchmark_version = benchmark_version

    @property
    def comparison_count(self) -> int:
        """Total number of completed comparisons."""
        return len(self._comparison_history)

    @property
    def rounds_completed(self) -> int:
        """Number of completed tournament rounds."""
        return sum(1 for r in self._rounds if r.completed)

    def register_models(self, model_ids: Sequence[str]) -> None:
        """Register models for the tournament.

        Args:
            model_ids: List of model identifiers.
        """
        for model_id in model_ids:
            if self._registry:
                entry = self._registry.get_model(model_id)
                if entry:
                    self.elo_calculator.ratings[model_id] = entry.rating
                else:
                    self._registry.register_model(model_id)
                    self.elo_calculator.ratings[model_id] = self._registry.get_rating(model_id)
            else:
                self.elo_calculator.ensure_registered(model_id)

    def get_recommended_rounds(self, num_models: int) -> int:
        """Get recommended number of tournament rounds.

        Uses Swiss-system formula: ceil(log2(n)) + 2

        Args:
            num_models: Number of models in tournament.

        Returns:
            Recommended number of rounds.
        """
        return self.elo_calculator.get_swiss_round_count(num_models)

    def schedule_round(
        self,
        prompt_ids: Sequence[str],
        round_number: int | None = None,
    ) -> TournamentRound:
        """Schedule comparisons for a tournament round.

        Args:
            prompt_ids: Prompts to use for this round.
            round_number: Round number (auto-incremented if None).

        Returns:
            TournamentRound with scheduled comparisons.
        """
        if round_number is None:
            round_number = len(self._rounds) + 1

        # Get models sorted by rating
        rankings = self.elo_calculator.get_rankings()
        model_ids = [model_id for model_id, _ in rankings]

        # Generate pairs based on strategy
        pairs = self._generate_pairs(model_ids)

        # Create scheduled comparisons
        comparisons = []
        for model_a, model_b in pairs:
            for prompt_id in prompt_ids:
                # Randomize position order
                position = self._rng.choice([PositionOrder.AB, PositionOrder.BA])
                comparison = ScheduledComparison(
                    model_a=model_a,
                    model_b=model_b,
                    prompt_id=prompt_id,
                    position_order=position,
                )

                # Skip if already completed
                if comparison.comparison_id not in self._completed_comparison_ids:
                    comparisons.append(comparison)

        tournament_round = TournamentRound(
            round_number=round_number,
            comparisons=comparisons,
        )
        self._rounds.append(tournament_round)
        return tournament_round

    def _generate_pairs(self, model_ids: list[str]) -> list[tuple[str, str]]:
        """Generate model pairs based on strategy.

        Args:
            model_ids: List of model IDs (sorted by rating for Swiss).

        Returns:
            List of (model_a, model_b) pairs.
        """
        if len(model_ids) < 2:
            return []

        if self.strategy == PairingStrategy.ROUND_ROBIN:
            return self._round_robin_pairs(model_ids)
        elif self.strategy == PairingStrategy.RANDOM:
            return self._random_pairs(model_ids)
        else:  # SWISS
            return self._swiss_pairs(model_ids)

    def _swiss_pairs(self, model_ids: list[str]) -> list[tuple[str, str]]:
        """Generate Swiss-system pairs (similar ratings paired together).

        Args:
            model_ids: Models sorted by rating (highest first).

        Returns:
            List of pairs.
        """
        pairs = []
        used = set()

        for i, model_a in enumerate(model_ids):
            if model_a in used:
                continue

            # Find closest-rated opponent not yet paired
            for model_b in model_ids[i + 1 :]:
                if model_b in used:
                    continue

                # Check if this pair has been compared too many times
                pair_key: tuple[str, str] = (min(model_a, model_b), max(model_a, model_b))
                pair_count = self._model_pair_counts.get(pair_key, 0)

                # Allow up to 3 comparisons per pair in Swiss
                if pair_count < 3:
                    pairs.append((model_a, model_b))
                    used.add(model_a)
                    used.add(model_b)
                    break

        return pairs

    def _round_robin_pairs(self, model_ids: list[str]) -> list[tuple[str, str]]:
        """Generate all possible pairs (round-robin).

        Args:
            model_ids: List of model IDs.

        Returns:
            List of all pairs.
        """
        pairs = []
        for i, model_a in enumerate(model_ids):
            for model_b in model_ids[i + 1 :]:
                pairs.append((model_a, model_b))
        return pairs

    def _random_pairs(self, model_ids: list[str]) -> list[tuple[str, str]]:
        """Generate random pairs.

        Args:
            model_ids: List of model IDs.

        Returns:
            List of random pairs.
        """
        shuffled = list(model_ids)
        self._rng.shuffle(shuffled)

        pairs = []
        for i in range(0, len(shuffled) - 1, 2):
            pairs.append((shuffled[i], shuffled[i + 1]))
        return pairs

    async def execute_comparison(
        self,
        scheduled: ScheduledComparison,
        model_clients: dict[str, BaseModelClient],
        judge: BaseJudge,
        prompt_text: str,
        system_prompt: str | None = None,
    ) -> ComparisonResult:
        """Execute a single comparison.

        Args:
            scheduled: Scheduled comparison to execute.
            model_clients: Dict mapping model_id to client.
            judge: Judge to evaluate responses.
            prompt_text: The actual prompt text.
            system_prompt: Optional system prompt.

        Returns:
            ComparisonResult with responses and verdict.

        Raises:
            KeyError: If model client not found.
        """
        # Get model clients
        client_a = model_clients[scheduled.model_a]
        client_b = model_clients[scheduled.model_b]

        # Generate responses from both models
        response_a = await client_a.generate(prompt_text, system_prompt=system_prompt)
        response_b = await client_b.generate(prompt_text, system_prompt=system_prompt)

        # Convert to data models
        response_a_data = ModelResponseData(
            text=response_a.text,
            tokens_used=response_a.tokens_used,
            latency_ms=response_a.latency_ms,
            model_id=response_a.model_id,
            timestamp=response_a.timestamp,
            cost_usd=response_a.cost_usd,
        )
        response_b_data = ModelResponseData(
            text=response_b.text,
            tokens_used=response_b.tokens_used,
            latency_ms=response_b.latency_ms,
            model_id=response_b.model_id,
            timestamp=response_b.timestamp,
            cost_usd=response_b.cost_usd,
        )

        # Determine presentation order based on position
        if scheduled.position_order == PositionOrder.AB:
            first_response = response_a_data.text
            second_response = response_b_data.text
        else:
            first_response = response_b_data.text
            second_response = response_a_data.text

        verdict = await judge.evaluate(
            prompt=prompt_text,
            response_a=first_response,
            response_b=second_response,
        )

        # Adjust verdict for position order
        adjusted_winner = self._adjust_winner_for_position(verdict.winner, scheduled.position_order)

        verdict_data = JudgeVerdictData(
            winner=adjusted_winner,
            confidence=verdict.confidence,
            reasoning=verdict.reasoning,
            raw_response=verdict.raw_response,
            latency_ms=verdict.latency_ms,
        )

        return ComparisonResult(
            scheduled=scheduled,
            response_a=response_a_data,
            response_b=response_b_data,
            verdict=verdict_data,
            judge_id=judge.model_id,
        )

    def _adjust_winner_for_position(
        self,
        winner: WinnerChoice,
        position: PositionOrder,
    ) -> WinnerChoice:
        """Adjust winner based on presentation position.

        If position was BA (B shown first), swap A/B in verdict.

        Args:
            winner: Original winner from judge.
            position: How responses were presented.

        Returns:
            Adjusted winner relative to original model_a/model_b.
        """
        if position == PositionOrder.AB:
            return winner

        # Position was BA, so swap
        if winner == WinnerChoice.A:
            return WinnerChoice.B
        elif winner == WinnerChoice.B:
            return WinnerChoice.A
        return winner  # Tie stays tie

    def record_result(self, result: ComparisonResult) -> None:
        """Record a comparison result and update ELO ratings.

        Args:
            result: Completed comparison result.
        """
        winner_literal: Literal["A", "B", "tie"]
        if result.verdict.winner == WinnerChoice.A:
            winner_literal = "A"
        elif result.verdict.winner == WinnerChoice.B:
            winner_literal = "B"
        else:
            winner_literal = "tie"

        self.elo_calculator.record_comparison(
            model_a=result.scheduled.model_a,
            model_b=result.scheduled.model_b,
            winner=winner_literal,
            prompt_id=result.scheduled.prompt_id,
        )

        if self._registry:
            self._registry.record_comparison(
                model_a=result.scheduled.model_a,
                model_b=result.scheduled.model_b,
                winner=winner_literal,
                prompt_id=result.scheduled.prompt_id,
                judge_id=result.judge_id,
                run_id=self._run_id,
                benchmark_version=self._benchmark_version,
            )

        self._completed_comparison_ids.add(result.scheduled.comparison_id)
        self._comparison_history.append(result)

        # Update pair count
        pair_key: tuple[str, str] = (
            min(result.scheduled.model_a, result.scheduled.model_b),
            max(result.scheduled.model_a, result.scheduled.model_b),
        )
        self._model_pair_counts[pair_key] = self._model_pair_counts.get(pair_key, 0) + 1

        # Update current round if applicable
        if self._rounds:
            current_round = self._rounds[-1]
            if not current_round.completed:
                current_round.results.append(result)
                if len(current_round.results) >= len(current_round.comparisons):
                    current_round.completed = True

    def get_rankings(self) -> list[tuple[str, float]]:
        """Get current model rankings.

        Returns:
            List of (model_id, rating) tuples sorted by rating.
        """
        return self.elo_calculator.get_rankings()

    def get_leaderboard(self) -> list[dict[str, Any]]:
        """Get formatted leaderboard.

        Returns:
            List of dicts with rank, model_id, rating, wins, losses, ties.
        """
        leaderboard = []
        for rank, (model_id, rating) in enumerate(self.get_rankings(), start=1):
            stats = self.elo_calculator.get_model_statistics(model_id)
            leaderboard.append(
                {
                    "rank": rank,
                    "model_id": model_id,
                    "rating": round(rating, 1),
                    "wins": stats.wins,
                    "losses": stats.losses,
                    "ties": stats.ties,
                    "win_rate": round(stats.win_rate, 3),
                }
            )
        return leaderboard

    def get_comparison_history(self) -> list[ComparisonRecordData]:
        """Get all comparison records.

        Returns:
            List of ComparisonRecordData for all comparisons.
        """
        return [result.to_record() for result in self._comparison_history]

    def get_position_bias_stats(self) -> dict[str, Any]:
        """Calculate position bias statistics.

        Returns:
            Dict with position bias metrics.
        """
        if not self._comparison_history:
            return {"total": 0, "a_wins": 0, "b_wins": 0, "ties": 0, "bias": 0.0}

        a_wins = sum(1 for r in self._comparison_history if r.verdict.winner == WinnerChoice.A)
        b_wins = sum(1 for r in self._comparison_history if r.verdict.winner == WinnerChoice.B)
        ties = sum(1 for r in self._comparison_history if r.verdict.winner == WinnerChoice.TIE)
        total = len(self._comparison_history)

        # Bias is deviation from 50% A-wins (excluding ties)
        non_tie = a_wins + b_wins
        bias = abs(a_wins / non_tie - 0.5) if non_tie > 0 else 0.0

        return {
            "total": total,
            "a_wins": a_wins,
            "b_wins": b_wins,
            "ties": ties,
            "bias": round(bias, 4),
        }

    def reset(self) -> None:
        """Reset all state."""
        self.elo_calculator.reset()
        self._completed_comparison_ids.clear()
        self._comparison_history.clear()
        self._rounds.clear()
        self._model_pair_counts.clear()


def create_pairwise_engine(
    strategy: str = "swiss",
    initial_rating: float = 1500.0,
    k_factor: float = 32.0,
    seed: int | None = None,
    registry: ELORegistry | None = None,
) -> PairwiseEngine:
    """Factory function to create a pairwise engine.

    Args:
        strategy: Pairing strategy ("swiss", "round_robin", "random").
        initial_rating: Starting ELO rating.
        k_factor: ELO K-factor.
        seed: Random seed for reproducibility.
        registry: Optional persistent ELO registry.

    Returns:
        Configured PairwiseEngine.
    """
    strategy_enum = PairingStrategy(strategy)
    return PairwiseEngine(
        initial_rating=initial_rating,
        k_factor=k_factor,
        strategy=strategy_enum,
        seed=seed,
        registry=registry,
    )
