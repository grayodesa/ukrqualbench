"""ELO rating calculator for pairwise comparisons.

Implements the standard ELO rating system used in chess,
adapted for LLM evaluation through pairwise comparisons.

References:
- https://en.wikipedia.org/wiki/Elo_rating_system
- Section 5.1 of UkrQualBench Technical Specification
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ELOCalculator:
    """ELO rating calculator for model comparisons.

    Attributes:
        initial_rating: Starting rating for all models (default: 1500).
        k_factor: How much ratings change per game (default: 32).
        ratings: Current ratings for all registered models.
        history: Full history of all rating updates.
    """

    initial_rating: float = 1500.0
    k_factor: float = 32.0
    ratings: dict[str, float] = field(default_factory=dict)
    history: list[RatingUpdate] = field(default_factory=list)

    def register_model(self, model_id: str) -> float:
        """Register a new model with initial rating.

        Args:
            model_id: Unique identifier for the model.

        Returns:
            The initial rating assigned.

        Raises:
            ValueError: If model is already registered.
        """
        if model_id in self.ratings:
            raise ValueError(f"Model {model_id} already registered")
        self.ratings[model_id] = self.initial_rating
        return self.initial_rating

    def ensure_registered(self, model_id: str) -> float:
        """Ensure model is registered, registering if needed.

        Args:
            model_id: Unique identifier for the model.

        Returns:
            Current rating (initial if just registered).
        """
        if model_id not in self.ratings:
            return self.register_model(model_id)
        return self.ratings[model_id]

    def get_rating(self, model_id: str) -> float:
        """Get current rating for a model.

        Args:
            model_id: Model identifier.

        Returns:
            Current ELO rating.

        Raises:
            KeyError: If model is not registered.
        """
        if model_id not in self.ratings:
            raise KeyError(f"Model {model_id} not registered")
        return self.ratings[model_id]

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score for player A against player B.

        The expected score is the probability of A winning plus
        half the probability of a draw.

        Args:
            rating_a: Rating of player A.
            rating_b: Rating of player B.

        Returns:
            Expected score for player A (between 0 and 1).
        """
        exponent = (rating_b - rating_a) / 400.0
        return 1.0 / (1.0 + math.pow(10, exponent))

    def update_rating(
        self, current_rating: float, expected: float, actual: float
    ) -> float:
        """Calculate new rating after a game.

        Args:
            current_rating: Current ELO rating.
            expected: Expected score (from expected_score()).
            actual: Actual score (1.0=win, 0.5=tie, 0.0=loss).

        Returns:
            New ELO rating.
        """
        return current_rating + self.k_factor * (actual - expected)

    def record_comparison(
        self,
        model_a: str,
        model_b: str,
        winner: Literal["A", "B", "tie"],
        *,
        prompt_id: str | None = None,
    ) -> RatingUpdate:
        """Record a pairwise comparison result and update ratings.

        Args:
            model_a: ID of first model.
            model_b: ID of second model.
            winner: "A" if model_a won, "B" if model_b won, "tie" for draw.
            prompt_id: Optional prompt ID for tracking.

        Returns:
            Rating update details.
        """
        # Ensure both models are registered
        self.ensure_registered(model_a)
        self.ensure_registered(model_b)

        # Get current ratings
        rating_a = self.ratings[model_a]
        rating_b = self.ratings[model_b]

        # Calculate expected scores
        expected_a = self.expected_score(rating_a, rating_b)
        expected_b = 1.0 - expected_a

        # Convert winner to actual scores
        if winner == "A":
            actual_a, actual_b = 1.0, 0.0
        elif winner == "B":
            actual_a, actual_b = 0.0, 1.0
        else:  # tie
            actual_a, actual_b = 0.5, 0.5

        # Calculate new ratings
        new_rating_a = self.update_rating(rating_a, expected_a, actual_a)
        new_rating_b = self.update_rating(rating_b, expected_b, actual_b)

        # Update stored ratings
        self.ratings[model_a] = new_rating_a
        self.ratings[model_b] = new_rating_b

        # Record update in history
        update = RatingUpdate(
            model_a=model_a,
            model_b=model_b,
            winner=winner,
            old_rating_a=rating_a,
            old_rating_b=rating_b,
            new_rating_a=new_rating_a,
            new_rating_b=new_rating_b,
            expected_a=expected_a,
            prompt_id=prompt_id,
        )
        self.history.append(update)

        return update

    def get_rankings(self) -> list[tuple[str, float]]:
        """Get models ranked by ELO rating (highest first).

        Returns:
            List of (model_id, rating) tuples sorted by rating.
        """
        return sorted(
            self.ratings.items(),
            key=lambda x: x[1],
            reverse=True,
        )

    def get_leaderboard(self) -> list[dict[str, float | str | int]]:
        """Get formatted leaderboard with ranks.

        Returns:
            List of dicts with rank, model_id, and rating.
        """
        rankings = self.get_rankings()
        return [
            {"rank": i + 1, "model_id": model_id, "rating": round(rating, 1)}
            for i, (model_id, rating) in enumerate(rankings)
        ]

    def get_model_statistics(self, model_id: str) -> ModelStatistics:
        """Get detailed statistics for a model.

        Args:
            model_id: Model identifier.

        Returns:
            Statistics including wins, losses, ties, and rating history.
        """
        if model_id not in self.ratings:
            raise KeyError(f"Model {model_id} not registered")

        wins = 0
        losses = 0
        ties = 0
        rating_history: list[float] = [self.initial_rating]

        for update in self.history:
            if update.model_a == model_id:
                if update.winner == "A":
                    wins += 1
                elif update.winner == "B":
                    losses += 1
                else:
                    ties += 1
                rating_history.append(update.new_rating_a)
            elif update.model_b == model_id:
                if update.winner == "B":
                    wins += 1
                elif update.winner == "A":
                    losses += 1
                else:
                    ties += 1
                rating_history.append(update.new_rating_b)

        return ModelStatistics(
            model_id=model_id,
            current_rating=self.ratings[model_id],
            wins=wins,
            losses=losses,
            ties=ties,
            total_games=wins + losses + ties,
            rating_history=rating_history,
        )

    def get_swiss_round_count(self, num_models: int) -> int:
        """Calculate number of rounds for Swiss-system tournament.

        Formula: ceil(log2(n)) + 2

        Args:
            num_models: Number of models in tournament.

        Returns:
            Recommended number of rounds.
        """
        if num_models < 2:
            return 0
        return math.ceil(math.log2(num_models)) + 2

    def reset(self) -> None:
        """Reset all ratings and history."""
        self.ratings.clear()
        self.history.clear()


@dataclass
class RatingUpdate:
    """Record of a single rating update."""

    model_a: str
    model_b: str
    winner: Literal["A", "B", "tie"]
    old_rating_a: float
    old_rating_b: float
    new_rating_a: float
    new_rating_b: float
    expected_a: float
    prompt_id: str | None = None

    @property
    def delta_a(self) -> float:
        """Rating change for model A."""
        return self.new_rating_a - self.old_rating_a

    @property
    def delta_b(self) -> float:
        """Rating change for model B."""
        return self.new_rating_b - self.old_rating_b


@dataclass
class ModelStatistics:
    """Detailed statistics for a model."""

    model_id: str
    current_rating: float
    wins: int
    losses: int
    ties: int
    total_games: int
    rating_history: list[float]

    @property
    def win_rate(self) -> float:
        """Proportion of games won (0-1)."""
        if self.total_games == 0:
            return 0.0
        return self.wins / self.total_games

    @property
    def rating_change(self) -> float:
        """Total rating change from initial."""
        if len(self.rating_history) < 2:
            return 0.0
        return self.rating_history[-1] - self.rating_history[0]
