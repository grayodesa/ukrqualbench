"""Tests for core evaluation engine components."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from ukrqualbench.core.evaluator import (
    BenchmarkTask,
    EvaluationConfig,
    EvaluationProgress,
    Evaluator,
    create_evaluator,
)
from ukrqualbench.core.pairwise import (
    ComparisonResult,
    PairingStrategy,
    PairwiseEngine,
    ScheduledComparison,
    TournamentRound,
    create_pairwise_engine,
)
from ukrqualbench.core.schemas import (
    ConfidenceLevel,
    JudgeVerdictData,
    ModelResponseData,
    PositionOrder,
    WinnerChoice,
)

# ============================================================
# Pairwise Engine Tests
# ============================================================


class TestPairingStrategy:
    """Tests for PairingStrategy enum."""

    def test_pairing_strategies(self):
        """Test all pairing strategies are defined."""
        assert PairingStrategy.SWISS.value == "swiss"
        assert PairingStrategy.ROUND_ROBIN.value == "round_robin"
        assert PairingStrategy.RANDOM.value == "random"

    def test_strategy_from_string(self):
        """Test creating strategy from string."""
        assert PairingStrategy("swiss") == PairingStrategy.SWISS
        assert PairingStrategy("round_robin") == PairingStrategy.ROUND_ROBIN


class TestScheduledComparison:
    """Tests for ScheduledComparison dataclass."""

    def test_scheduled_comparison_creation(self):
        """Test creating scheduled comparison."""
        scheduled = ScheduledComparison(
            model_a="model_a",
            model_b="model_b",
            prompt_id="prompt_1",
            position_order=PositionOrder.AB,
        )

        assert scheduled.model_a == "model_a"
        assert scheduled.model_b == "model_b"
        assert scheduled.prompt_id == "prompt_1"
        assert scheduled.position_order == PositionOrder.AB
        assert len(scheduled.comparison_id) == 16  # SHA256 truncated

    def test_comparison_id_deterministic(self):
        """Test that comparison ID is deterministic."""
        scheduled1 = ScheduledComparison(
            model_a="model_a",
            model_b="model_b",
            prompt_id="prompt_1",
            position_order=PositionOrder.AB,
        )
        scheduled2 = ScheduledComparison(
            model_a="model_a",
            model_b="model_b",
            prompt_id="prompt_1",
            position_order=PositionOrder.BA,
        )
        # Same models and prompt should have same ID
        assert scheduled1.comparison_id == scheduled2.comparison_id

    def test_comparison_id_order_independent(self):
        """Test that comparison ID is order-independent."""
        scheduled1 = ScheduledComparison(
            model_a="model_a",
            model_b="model_b",
            prompt_id="prompt_1",
            position_order=PositionOrder.AB,
        )
        scheduled2 = ScheduledComparison(
            model_a="model_b",
            model_b="model_a",
            prompt_id="prompt_1",
            position_order=PositionOrder.AB,
        )
        # Swapped order should have same ID
        assert scheduled1.comparison_id == scheduled2.comparison_id


class TestTournamentRound:
    """Tests for TournamentRound dataclass."""

    def test_tournament_round_creation(self):
        """Test creating tournament round."""
        round_ = TournamentRound(round_number=1)
        assert round_.round_number == 1
        assert round_.comparisons == []
        assert round_.results == []
        assert not round_.completed

    def test_tournament_round_progress(self):
        """Test round progress calculation."""
        scheduled = ScheduledComparison(
            model_a="a", model_b="b", prompt_id="p1", position_order=PositionOrder.AB
        )
        round_ = TournamentRound(round_number=1, comparisons=[scheduled, scheduled])

        assert round_.progress == 0.0

        # Add one result
        result = MagicMock()
        round_.results.append(result)
        assert round_.progress == 0.5

        # Add second result
        round_.results.append(result)
        assert round_.progress == 1.0

    def test_empty_round_progress(self):
        """Test progress of empty round."""
        round_ = TournamentRound(round_number=1)
        assert round_.progress == 1.0  # No comparisons = complete


class TestPairwiseEngine:
    """Tests for PairwiseEngine class."""

    def test_engine_creation(self):
        """Test creating pairwise engine."""
        engine = PairwiseEngine()
        assert engine.strategy == PairingStrategy.SWISS
        assert engine.comparison_count == 0
        assert engine.rounds_completed == 0

    def test_engine_with_custom_params(self):
        """Test engine with custom parameters."""
        engine = PairwiseEngine(
            initial_rating=1600.0,
            k_factor=24.0,
            strategy=PairingStrategy.ROUND_ROBIN,
            seed=42,
        )
        assert engine.strategy == PairingStrategy.ROUND_ROBIN

    def test_register_models(self):
        """Test registering models."""
        engine = PairwiseEngine()
        engine.register_models(["model_a", "model_b", "model_c"])
        assert len(engine.ratings) == 3
        assert all(r == 1500.0 for r in engine.ratings.values())

    def test_get_recommended_rounds(self):
        """Test recommended rounds calculation."""
        engine = PairwiseEngine()
        engine.register_models(["m1", "m2", "m3", "m4"])

        rounds = engine.get_recommended_rounds(4)
        # Swiss formula: ceil(log2(4)) + 2 = 2 + 2 = 4
        assert rounds == 4

    def test_schedule_round_swiss(self):
        """Test scheduling a round with Swiss pairing."""
        engine = PairwiseEngine(seed=42)
        engine.register_models(["m1", "m2", "m3", "m4"])

        round_ = engine.schedule_round(["p1", "p2"], round_number=1)

        assert round_.round_number == 1
        # Swiss distributes prompts across all possible pairs
        # 4 models = 6 possible pairs, 2 prompts = 2 comparisons (one per prompt)
        assert len(round_.comparisons) == 2

    def test_schedule_round_round_robin(self):
        """Test scheduling a round with round-robin pairing."""
        engine = PairwiseEngine(strategy=PairingStrategy.ROUND_ROBIN)
        engine.register_models(["m1", "m2", "m3"])

        round_ = engine.schedule_round(["p1"], round_number=1)

        # Round robin distributes prompts across all possible pairs
        # 3 models = 3 possible pairs, 1 prompt = 1 comparison
        assert len(round_.comparisons) == 1

    def test_schedule_round_auto_number(self):
        """Test automatic round numbering."""
        engine = PairwiseEngine()
        engine.register_models(["m1", "m2"])

        round1 = engine.schedule_round(["p1"])
        assert round1.round_number == 1

        round2 = engine.schedule_round(["p2"])
        assert round2.round_number == 2

    def test_record_result(self):
        """Test recording comparison results."""
        engine = PairwiseEngine()
        engine.register_models(["m1", "m2"])

        scheduled = ScheduledComparison(
            model_a="m1", model_b="m2", prompt_id="p1", position_order=PositionOrder.AB
        )

        response_a = ModelResponseData(
            text="Response A",
            tokens_used=10,
            latency_ms=100,
            model_id="m1",
            timestamp=datetime.now(),
            cost_usd=0.001,
        )
        response_b = ModelResponseData(
            text="Response B",
            tokens_used=10,
            latency_ms=100,
            model_id="m2",
            timestamp=datetime.now(),
            cost_usd=0.001,
        )
        verdict = JudgeVerdictData(
            winner=WinnerChoice.A,
            confidence=ConfidenceLevel.HIGH,
            reasoning="A is better",
            raw_response="A",
            latency_ms=50,
        )

        result = ComparisonResult(
            scheduled=scheduled,
            response_a=response_a,
            response_b=response_b,
            verdict=verdict,
            judge_id="judge",
        )

        engine.record_result(result)

        assert engine.comparison_count == 1
        # Rating should have changed
        ratings = engine.ratings
        assert ratings["m1"] > 1500.0  # Winner
        assert ratings["m2"] < 1500.0  # Loser

    def test_get_rankings(self):
        """Test getting rankings."""
        engine = PairwiseEngine()
        engine.register_models(["m1", "m2", "m3"])

        rankings = engine.get_rankings()
        assert len(rankings) == 3
        # All equal initially
        for _model_id, rating in rankings:
            assert rating == 1500.0

    def test_get_leaderboard(self):
        """Test getting leaderboard."""
        engine = PairwiseEngine()
        engine.register_models(["m1", "m2"])

        leaderboard = engine.get_leaderboard()
        assert len(leaderboard) == 2
        assert leaderboard[0]["rank"] == 1
        assert leaderboard[1]["rank"] == 2
        assert "model_id" in leaderboard[0]
        assert "rating" in leaderboard[0]
        assert "wins" in leaderboard[0]

    def test_position_bias_stats_empty(self):
        """Test position bias with no comparisons."""
        engine = PairwiseEngine()
        stats = engine.get_position_bias_stats()
        assert stats["total"] == 0
        assert stats["bias"] == 0.0

    def test_comparison_history(self):
        """Test getting comparison history."""
        engine = PairwiseEngine()
        engine.register_models(["m1", "m2"])

        # Initially empty
        history = engine.get_comparison_history()
        assert history == []

    def test_reset(self):
        """Test resetting engine state."""
        engine = PairwiseEngine()
        engine.register_models(["m1", "m2"])
        engine.schedule_round(["p1"])

        engine.reset()
        assert engine.comparison_count == 0
        assert engine.rounds_completed == 0


class TestCreatePairwiseEngine:
    """Tests for create_pairwise_engine factory function."""

    def test_factory_default(self):
        """Test factory with defaults."""
        engine = create_pairwise_engine()
        assert engine.strategy == PairingStrategy.SWISS

    def test_factory_custom_strategy(self):
        """Test factory with custom strategy."""
        engine = create_pairwise_engine(strategy="round_robin")
        assert engine.strategy == PairingStrategy.ROUND_ROBIN

    def test_factory_with_seed(self):
        """Test factory with seed."""
        engine = create_pairwise_engine(seed=42)
        # Should not raise
        assert engine is not None


# ============================================================
# Evaluator Tests
# ============================================================


class TestBenchmarkTask:
    """Tests for BenchmarkTask dataclass."""

    def test_task_creation(self):
        """Test creating benchmark task."""
        task = BenchmarkTask(
            id="task_1",
            type="multiple_choice",
            category="grammar",
            prompt="Choose the correct option",
            reference="A",
        )

        assert task.id == "task_1"
        assert task.type == "multiple_choice"
        assert task.category == "grammar"
        assert task.prompt == "Choose the correct option"
        assert task.reference == "A"
        assert task.metadata == {}

    def test_task_with_metadata(self):
        """Test task with metadata."""
        task = BenchmarkTask(
            id="task_1",
            type="generation",
            category="creative",
            prompt="Write a poem",
            metadata={"difficulty": "hard", "topic": "nature"},
        )

        assert task.metadata["difficulty"] == "hard"
        assert task.metadata["topic"] == "nature"


class TestEvaluationProgress:
    """Tests for EvaluationProgress dataclass."""

    def test_progress_creation(self):
        """Test creating progress object."""
        progress = EvaluationProgress()
        assert progress.total_tasks == 0
        assert progress.completed_tasks == 0
        assert progress.errors == 0

    def test_progress_percent(self):
        """Test progress percentage calculation."""
        progress = EvaluationProgress(
            total_comparisons=100,
            completed_comparisons=50,
        )
        assert progress.progress_percent == 50.0

    def test_progress_percent_zero_total(self):
        """Test progress with zero total."""
        progress = EvaluationProgress()
        assert progress.progress_percent == 0.0

    def test_elapsed_minutes(self):
        """Test elapsed time calculation."""
        import time

        start = time.time()
        progress = EvaluationProgress(start_time=start - 120)  # 2 minutes ago
        assert abs(progress.elapsed_minutes - 2.0) < 0.1

    def test_estimated_remaining(self):
        """Test estimated remaining time."""
        import time

        progress = EvaluationProgress(
            total_comparisons=100,
            completed_comparisons=50,
            start_time=time.time() - 60,  # 1 minute ago
        )
        # 50 done in 1 minute, 50 remaining should take ~1 more minute
        assert 0.9 < progress.estimated_remaining_minutes < 1.1


class TestEvaluationConfig:
    """Tests for EvaluationConfig dataclass."""

    def test_config_defaults(self):
        """Test default config values."""
        config = EvaluationConfig()
        assert config.benchmark_version == "base"
        assert config.max_concurrent == 5
        assert config.checkpoint_interval == 50
        assert config.budget_limit_usd is None
        assert config.auto_resume is True

    def test_config_custom(self):
        """Test custom config values."""
        config = EvaluationConfig(
            benchmark_version="lite",
            max_concurrent=3,
            budget_limit_usd=10.0,
        )
        assert config.benchmark_version == "lite"
        assert config.max_concurrent == 3
        assert config.budget_limit_usd == 10.0


class TestEvaluator:
    """Tests for Evaluator class."""

    def test_evaluator_creation(self):
        """Test creating evaluator."""
        evaluator = Evaluator()
        assert evaluator is not None

    def test_add_model(self):
        """Test adding a model."""
        evaluator = Evaluator()
        mock_client = MagicMock()
        evaluator.add_model("test-model", mock_client)

        assert "test-model" in evaluator._model_clients

    def test_set_judge(self):
        """Test setting a judge."""
        evaluator = Evaluator()
        mock_judge = MagicMock()
        mock_judge.model_id = "judge-model"
        evaluator.set_judge(mock_judge)

        assert evaluator._judge is mock_judge

    def test_add_detector(self):
        """Test adding a detector."""
        evaluator = Evaluator()
        mock_detector = MagicMock()
        evaluator.add_detector("russism", mock_detector)

        assert "russism" in evaluator._detectors

    def test_load_tasks(self):
        """Test loading tasks."""
        evaluator = Evaluator()
        tasks = [
            BenchmarkTask(id="1", type="generation", category="test", prompt="Test"),
            BenchmarkTask(id="2", type="generation", category="test", prompt="Test2"),
        ]
        evaluator.load_tasks(tasks)

        assert len(evaluator._tasks) == 2
        assert evaluator._progress.total_tasks == 2

    def test_set_progress_callback(self):
        """Test setting progress callback."""
        evaluator = Evaluator()
        callback = MagicMock()
        evaluator.set_progress_callback(callback)

        assert evaluator._on_progress is callback

    def test_get_progress(self):
        """Test getting progress."""
        evaluator = Evaluator()
        progress = evaluator.get_progress()

        assert isinstance(progress, EvaluationProgress)

    def test_get_current_rankings(self):
        """Test getting current rankings."""
        evaluator = Evaluator()
        mock_client = MagicMock()
        evaluator.add_model("model1", mock_client)
        evaluator.add_model("model2", mock_client)

        rankings = evaluator.get_current_rankings()
        assert len(rankings) == 2

    def test_get_statistics(self):
        """Test getting statistics."""
        evaluator = Evaluator()
        mock_client = MagicMock()
        evaluator.add_model("model1", mock_client)

        stats = evaluator.get_statistics()
        assert "run_id" in stats
        assert "models" in stats
        assert "progress" in stats
        assert "cost_usd" in stats

    @pytest.mark.asyncio
    async def test_run_no_models_error(self):
        """Test run raises error without models."""
        evaluator = Evaluator()
        mock_judge = MagicMock()
        evaluator.set_judge(mock_judge)

        with pytest.raises(RuntimeError, match="No models configured"):
            await evaluator.run()

    @pytest.mark.asyncio
    async def test_run_no_judge_error(self):
        """Test run raises error without judge."""
        evaluator = Evaluator()
        mock_client = MagicMock()
        evaluator.add_model("model", mock_client)

        with pytest.raises(RuntimeError, match="No judge configured"):
            await evaluator.run()

    def test_provider_detection(self):
        """Test provider detection from model ID."""
        evaluator = Evaluator()

        assert evaluator._get_provider("gpt-4o") == "openai"
        assert evaluator._get_provider("claude-3.5-sonnet") == "anthropic"
        assert evaluator._get_provider("gemini-pro") == "google"
        assert evaluator._get_provider("org/model") == "nebius"
        assert evaluator._get_provider("unknown-model") == "unknown"


class TestCreateEvaluator:
    """Tests for create_evaluator factory function."""

    def test_factory_default(self):
        """Test factory with defaults."""
        evaluator = create_evaluator()
        assert evaluator._eval_config.benchmark_version == "base"

    def test_factory_custom(self):
        """Test factory with custom params."""
        evaluator = create_evaluator(
            benchmark_version="lite",
            max_concurrent=3,
            budget_limit_usd=5.0,
        )
        assert evaluator._eval_config.benchmark_version == "lite"
        assert evaluator._eval_config.max_concurrent == 3
        assert evaluator._eval_config.budget_limit_usd == 5.0

    def test_factory_with_output_dir(self, tmp_path):
        """Test factory with output directory."""
        evaluator = create_evaluator(output_dir=tmp_path / "results")
        assert evaluator._output_dir.exists()


# ============================================================
# Integration Tests
# ============================================================


class TestPairwiseEngineIntegration:
    """Integration tests for pairwise engine."""

    def test_full_tournament_workflow(self):
        """Test complete tournament workflow."""
        engine = PairwiseEngine(seed=42)
        engine.register_models(["m1", "m2", "m3", "m4"])

        # Run multiple rounds
        prompts = ["p1", "p2"]
        num_rounds = engine.get_recommended_rounds(4)

        for round_num in range(1, num_rounds + 1):
            round_ = engine.schedule_round(prompts, round_number=round_num)

            # Simulate results
            for scheduled in round_.comparisons:
                response = ModelResponseData(
                    text="Test",
                    tokens_used=10,
                    latency_ms=100,
                    model_id=scheduled.model_a,
                    timestamp=datetime.now(),
                    cost_usd=0.001,
                )
                verdict = JudgeVerdictData(
                    winner=WinnerChoice.A,  # Always A wins for test
                    confidence=ConfidenceLevel.HIGH,
                    reasoning="",
                    raw_response="",
                    latency_ms=50,
                )
                result = ComparisonResult(
                    scheduled=scheduled,
                    response_a=response,
                    response_b=response,
                    verdict=verdict,
                    judge_id="judge",
                )
                engine.record_result(result)

        # Verify tournament completed
        assert engine.comparison_count > 0
        rankings = engine.get_rankings()
        assert len(rankings) == 4

    def test_position_bias_tracking(self):
        """Test that position bias is tracked."""
        engine = PairwiseEngine(seed=42)
        engine.register_models(["m1", "m2"])

        # Create comparisons with both positions
        for position in [PositionOrder.AB, PositionOrder.BA]:
            scheduled = ScheduledComparison(
                model_a="m1",
                model_b="m2",
                prompt_id=f"p_{position.value}",
                position_order=position,
            )
            response = ModelResponseData(
                text="Test",
                tokens_used=10,
                latency_ms=100,
                model_id="m1",
                timestamp=datetime.now(),
                cost_usd=0.001,
            )
            verdict = JudgeVerdictData(
                winner=WinnerChoice.A,
                confidence=ConfidenceLevel.HIGH,
                reasoning="",
                raw_response="",
                latency_ms=50,
            )
            result = ComparisonResult(
                scheduled=scheduled,
                response_a=response,
                response_b=response,
                verdict=verdict,
                judge_id="judge",
            )
            engine.record_result(result)

        stats = engine.get_position_bias_stats()
        assert stats["total"] == 2


class TestComparisonResultConversion:
    """Tests for ComparisonResult to record conversion."""

    def test_to_record(self):
        """Test converting result to record."""
        scheduled = ScheduledComparison(
            model_a="m1", model_b="m2", prompt_id="p1", position_order=PositionOrder.AB
        )
        response_a = ModelResponseData(
            text="A",
            tokens_used=10,
            latency_ms=100,
            model_id="m1",
            timestamp=datetime.now(),
            cost_usd=0.001,
        )
        response_b = ModelResponseData(
            text="B",
            tokens_used=10,
            latency_ms=100,
            model_id="m2",
            timestamp=datetime.now(),
            cost_usd=0.001,
        )
        verdict = JudgeVerdictData(
            winner=WinnerChoice.A,
            confidence=ConfidenceLevel.HIGH,
            reasoning="Better",
            raw_response="A",
            latency_ms=50,
        )

        result = ComparisonResult(
            scheduled=scheduled,
            response_a=response_a,
            response_b=response_b,
            verdict=verdict,
            judge_id="judge",
        )

        record = result.to_record()

        assert record.comparison_id == scheduled.comparison_id
        assert record.prompt_id == "p1"
        assert record.model_a_id == "m1"
        assert record.model_b_id == "m2"
        assert record.judge_id == "judge"

    def test_winner_model_id(self):
        """Test getting winner model ID."""
        scheduled = ScheduledComparison(
            model_a="m1", model_b="m2", prompt_id="p1", position_order=PositionOrder.AB
        )
        response = ModelResponseData(
            text="Test",
            tokens_used=10,
            latency_ms=100,
            model_id="m1",
            timestamp=datetime.now(),
            cost_usd=0.001,
        )

        # Test A wins
        verdict_a = JudgeVerdictData(
            winner=WinnerChoice.A,
            confidence=ConfidenceLevel.HIGH,
            reasoning="",
            raw_response="",
            latency_ms=50,
        )
        result_a = ComparisonResult(
            scheduled=scheduled,
            response_a=response,
            response_b=response,
            verdict=verdict_a,
            judge_id="j",
        )
        assert result_a.winner_model_id == "m1"

        # Test B wins
        verdict_b = JudgeVerdictData(
            winner=WinnerChoice.B,
            confidence=ConfidenceLevel.HIGH,
            reasoning="",
            raw_response="",
            latency_ms=50,
        )
        result_b = ComparisonResult(
            scheduled=scheduled,
            response_a=response,
            response_b=response,
            verdict=verdict_b,
            judge_id="j",
        )
        assert result_b.winner_model_id == "m2"

        # Test tie
        verdict_tie = JudgeVerdictData(
            winner=WinnerChoice.TIE,
            confidence=ConfidenceLevel.HIGH,
            reasoning="",
            raw_response="",
            latency_ms=50,
        )
        result_tie = ComparisonResult(
            scheduled=scheduled,
            response_a=response,
            response_b=response,
            verdict=verdict_tie,
            judge_id="j",
        )
        assert result_tie.winner_model_id is None
