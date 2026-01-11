"""Tests for ELO rating calculator and persistent registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from ukrqualbench.core.elo import ELOCalculator, ModelStatistics, RatingUpdate
from ukrqualbench.core.elo_registry import ELORegistry


class TestELOCalculator:
    """Tests for ELOCalculator class."""

    def test_initial_rating(self, elo_calculator: ELOCalculator) -> None:
        """Test that models start with initial rating."""
        rating = elo_calculator.register_model("model_a")
        assert rating == 1500.0

    def test_double_registration_raises(self, elo_calculator: ELOCalculator) -> None:
        """Test that registering same model twice raises."""
        elo_calculator.register_model("model_a")
        with pytest.raises(ValueError, match="already registered"):
            elo_calculator.register_model("model_a")

    def test_ensure_registered_creates(self, elo_calculator: ELOCalculator) -> None:
        """Test ensure_registered creates new model."""
        rating = elo_calculator.ensure_registered("model_a")
        assert rating == 1500.0
        assert "model_a" in elo_calculator.ratings

    def test_ensure_registered_existing(self, elo_calculator: ELOCalculator) -> None:
        """Test ensure_registered returns existing rating."""
        elo_calculator.register_model("model_a")
        elo_calculator.ratings["model_a"] = 1600.0
        rating = elo_calculator.ensure_registered("model_a")
        assert rating == 1600.0

    def test_expected_score_equal_ratings(self, elo_calculator: ELOCalculator) -> None:
        """Test expected score with equal ratings."""
        expected = elo_calculator.expected_score(1500, 1500)
        assert expected == pytest.approx(0.5)

    def test_expected_score_higher_rating(self, elo_calculator: ELOCalculator) -> None:
        """Test expected score when A is higher rated."""
        expected = elo_calculator.expected_score(1600, 1400)
        assert expected > 0.5  # A should be expected to win

    def test_expected_score_lower_rating(self, elo_calculator: ELOCalculator) -> None:
        """Test expected score when A is lower rated."""
        expected = elo_calculator.expected_score(1400, 1600)
        assert expected < 0.5  # A should be expected to lose

    def test_expected_score_400_point_diff(self, elo_calculator: ELOCalculator) -> None:
        """Test expected score with 400 point difference."""
        expected = elo_calculator.expected_score(1900, 1500)
        assert expected == pytest.approx(0.909, rel=0.01)

    def test_update_rating_win(self, elo_calculator: ELOCalculator) -> None:
        """Test rating update on win."""
        # Equal ratings, A wins
        new_rating = elo_calculator.update_rating(1500, 0.5, 1.0)
        assert new_rating > 1500  # Rating should increase

    def test_update_rating_loss(self, elo_calculator: ELOCalculator) -> None:
        """Test rating update on loss."""
        new_rating = elo_calculator.update_rating(1500, 0.5, 0.0)
        assert new_rating < 1500  # Rating should decrease

    def test_update_rating_tie(self, elo_calculator: ELOCalculator) -> None:
        """Test rating update on tie with equal ratings."""
        new_rating = elo_calculator.update_rating(1500, 0.5, 0.5)
        assert new_rating == pytest.approx(1500)  # No change on tie

    def test_record_comparison_win_a(self, elo_calculator: ELOCalculator) -> None:
        """Test recording comparison where A wins."""
        update = elo_calculator.record_comparison("model_a", "model_b", "A")

        assert update.winner == "A"
        assert update.new_rating_a > update.old_rating_a
        assert update.new_rating_b < update.old_rating_b
        assert elo_calculator.ratings["model_a"] > elo_calculator.ratings["model_b"]

    def test_record_comparison_win_b(self, elo_calculator: ELOCalculator) -> None:
        """Test recording comparison where B wins."""
        update = elo_calculator.record_comparison("model_a", "model_b", "B")

        assert update.winner == "B"
        assert update.new_rating_a < update.old_rating_a
        assert update.new_rating_b > update.old_rating_b
        assert elo_calculator.ratings["model_b"] > elo_calculator.ratings["model_a"]

    def test_record_comparison_tie(self, elo_calculator: ELOCalculator) -> None:
        """Test recording comparison with tie."""
        update = elo_calculator.record_comparison("model_a", "model_b", "tie")

        assert update.winner == "tie"
        # Equal starting ratings, tie should result in no change
        assert update.new_rating_a == pytest.approx(1500)
        assert update.new_rating_b == pytest.approx(1500)

    def test_record_comparison_history(self, elo_calculator: ELOCalculator) -> None:
        """Test that comparisons are recorded in history."""
        elo_calculator.record_comparison("model_a", "model_b", "A")
        elo_calculator.record_comparison("model_a", "model_c", "B")

        assert len(elo_calculator.history) == 2
        assert elo_calculator.history[0].winner == "A"
        assert elo_calculator.history[1].winner == "B"

    def test_get_rankings(self, elo_calculator: ELOCalculator) -> None:
        """Test rankings are sorted correctly."""
        elo_calculator.register_model("model_a")
        elo_calculator.register_model("model_b")
        elo_calculator.register_model("model_c")

        # A wins against B
        elo_calculator.record_comparison("model_a", "model_b", "A")
        # A wins against C
        elo_calculator.record_comparison("model_a", "model_c", "A")

        rankings = elo_calculator.get_rankings()

        # A should be first
        assert rankings[0][0] == "model_a"
        assert rankings[0][1] > 1500

    def test_get_leaderboard(self, elo_calculator: ELOCalculator) -> None:
        """Test leaderboard format."""
        elo_calculator.register_model("model_a")
        elo_calculator.register_model("model_b")

        leaderboard = elo_calculator.get_leaderboard()

        assert len(leaderboard) == 2
        assert leaderboard[0]["rank"] == 1
        assert "model_id" in leaderboard[0]
        assert "rating" in leaderboard[0]

    def test_get_model_statistics(self, elo_calculator: ELOCalculator) -> None:
        """Test model statistics calculation."""
        elo_calculator.register_model("model_a")
        elo_calculator.register_model("model_b")

        # A wins twice
        elo_calculator.record_comparison("model_a", "model_b", "A")
        elo_calculator.record_comparison("model_a", "model_b", "A")
        # A loses once
        elo_calculator.record_comparison("model_a", "model_b", "B")

        stats = elo_calculator.get_model_statistics("model_a")

        assert stats.wins == 2
        assert stats.losses == 1
        assert stats.ties == 0
        assert stats.total_games == 3

    def test_swiss_round_count(self, elo_calculator: ELOCalculator) -> None:
        """Test Swiss tournament round calculation."""
        # 8 models: ceil(log2(8)) + 2 = 3 + 2 = 5
        assert elo_calculator.get_swiss_round_count(8) == 5

        # 10 models: ceil(log2(10)) + 2 = 4 + 2 = 6
        assert elo_calculator.get_swiss_round_count(10) == 6

        # 2 models: ceil(log2(2)) + 2 = 1 + 2 = 3
        assert elo_calculator.get_swiss_round_count(2) == 3

        # 1 model: 0 (no tournament needed)
        assert elo_calculator.get_swiss_round_count(1) == 0

    def test_reset(self, elo_calculator: ELOCalculator) -> None:
        """Test reset clears all data."""
        elo_calculator.register_model("model_a")
        elo_calculator.register_model("model_b")
        elo_calculator.record_comparison("model_a", "model_b", "A")

        elo_calculator.reset()

        assert len(elo_calculator.ratings) == 0
        assert len(elo_calculator.history) == 0


class TestRatingUpdate:
    """Tests for RatingUpdate dataclass."""

    def test_delta_calculation(self) -> None:
        """Test delta properties."""
        update = RatingUpdate(
            model_a="a",
            model_b="b",
            winner="A",
            old_rating_a=1500.0,
            old_rating_b=1500.0,
            new_rating_a=1516.0,
            new_rating_b=1484.0,
            expected_a=0.5,
        )

        assert update.delta_a == pytest.approx(16.0)
        assert update.delta_b == pytest.approx(-16.0)


class TestModelStatistics:
    """Tests for ModelStatistics dataclass."""

    def test_win_rate(self) -> None:
        """Test win rate calculation."""
        stats = ModelStatistics(
            model_id="test",
            current_rating=1600.0,
            wins=6,
            losses=3,
            ties=1,
            total_games=10,
            rating_history=[1500, 1520, 1540, 1560, 1580, 1600],
        )

        assert stats.win_rate == pytest.approx(0.6)

    def test_win_rate_zero_games(self) -> None:
        """Test win rate with no games."""
        stats = ModelStatistics(
            model_id="test",
            current_rating=1500.0,
            wins=0,
            losses=0,
            ties=0,
            total_games=0,
            rating_history=[1500],
        )

        assert stats.win_rate == 0.0

    def test_rating_change(self) -> None:
        """Test rating change calculation."""
        stats = ModelStatistics(
            model_id="test",
            current_rating=1600.0,
            wins=5,
            losses=2,
            ties=0,
            total_games=7,
            rating_history=[1500, 1516, 1532, 1548, 1564, 1580, 1596, 1600],
        )

        assert stats.rating_change == pytest.approx(100.0)


class TestELORegistry:
    """Tests for persistent ELO registry."""

    @pytest.fixture
    def temp_registry_path(self, tmp_path: Path) -> Path:
        return tmp_path / "test_registry.json"

    @pytest.fixture
    def registry(self, temp_registry_path: Path) -> ELORegistry:
        return ELORegistry(registry_path=temp_registry_path)

    def test_register_model(self, registry: ELORegistry) -> None:
        entry = registry.register_model("gpt-4o")
        assert entry.model_id == "gpt-4o"
        assert entry.rating == 1500.0
        assert entry.games_played == 0
        assert entry.is_provisional is True

    def test_register_model_returns_existing(self, registry: ELORegistry) -> None:
        registry.register_model("gpt-4o")
        registry.models["gpt-4o"].rating = 1600.0
        entry = registry.register_model("gpt-4o")
        assert entry.rating == 1600.0

    def test_record_comparison_updates_ratings(self, registry: ELORegistry) -> None:
        registry.record_comparison("model_a", "model_b", "A")

        assert registry.models["model_a"].rating > 1500
        assert registry.models["model_b"].rating < 1500
        assert registry.models["model_a"].wins == 1
        assert registry.models["model_b"].losses == 1

    def test_record_comparison_tie(self, registry: ELORegistry) -> None:
        registry.record_comparison("model_a", "model_b", "tie")

        assert registry.models["model_a"].rating == pytest.approx(1500)
        assert registry.models["model_a"].ties == 1
        assert registry.models["model_b"].ties == 1

    def test_save_and_load(self, temp_registry_path: Path) -> None:
        registry = ELORegistry(registry_path=temp_registry_path)
        registry.record_comparison("model_a", "model_b", "A")
        registry.record_comparison("model_a", "model_c", "B")
        registry.save()

        loaded = ELORegistry(registry_path=temp_registry_path)
        assert len(loaded.models) == 3
        assert loaded.models["model_a"].wins == 1
        assert loaded.models["model_a"].losses == 1
        assert loaded.comparison_count == 2

    def test_get_rankings(self, registry: ELORegistry) -> None:
        registry.record_comparison("model_a", "model_b", "A")
        registry.record_comparison("model_a", "model_c", "A")

        rankings = registry.get_rankings()
        assert rankings[0][0] == "model_a"
        assert rankings[0][1] > 1500

    def test_get_leaderboard(self, registry: ELORegistry) -> None:
        registry.record_comparison("model_a", "model_b", "A")

        leaderboard = registry.get_leaderboard()
        assert len(leaderboard) == 2
        assert leaderboard[0]["rank"] == 1
        assert "wins" in leaderboard[0]
        assert "provisional" in leaderboard[0]

    def test_get_anchor_models_prefers_non_provisional(self, registry: ELORegistry) -> None:
        for _ in range(35):
            registry.record_comparison("established", "other", "A")
        registry.register_model("new_model")

        anchors = registry.get_anchor_models(2)
        assert "established" in anchors
        assert "new_model" not in anchors

    def test_get_new_models(self, registry: ELORegistry) -> None:
        registry.register_model("existing")
        new = registry.get_new_models(["existing", "brand_new"])
        assert new == ["brand_new"]

    def test_get_existing_models(self, registry: ELORegistry) -> None:
        registry.register_model("existing")
        existing = registry.get_existing_models(["existing", "brand_new"])
        assert existing == ["existing"]

    def test_get_head_to_head(self, registry: ELORegistry) -> None:
        registry.record_comparison("a", "b", "A")
        registry.record_comparison("a", "b", "A")
        registry.record_comparison("a", "b", "B")

        h2h = registry.get_head_to_head("a", "b")
        assert h2h["a_wins"] == 2
        assert h2h["b_wins"] == 1
        assert h2h["ties"] == 0

    def test_provisional_threshold(self, registry: ELORegistry) -> None:
        for i in range(29):
            registry.record_comparison("model_a", f"opp_{i}", "A")
        assert registry.models["model_a"].is_provisional is True

        registry.record_comparison("model_a", "opp_30", "A")
        assert registry.models["model_a"].is_provisional is False

    def test_reset(self, registry: ELORegistry) -> None:
        registry.record_comparison("a", "b", "A")
        registry.reset()

        assert len(registry.models) == 0
        assert registry.comparison_count == 0

    def test_contains(self, registry: ELORegistry) -> None:
        registry.register_model("gpt-4o")
        assert "gpt-4o" in registry
        assert "claude" not in registry
