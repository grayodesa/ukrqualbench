"""Tests for checkpoint system."""

from __future__ import annotations

from pathlib import Path

import pytest

from ukrqualbench.core.checkpoint import (
    CheckpointData,
    CheckpointManager,
    ComparisonResult,
    create_initial_checkpoint,
    generate_comparison_key,
    generate_run_id,
)


class TestGenerateComparisonKey:
    """Tests for comparison key generation."""

    def test_deterministic(self) -> None:
        """Test key is deterministic."""
        key1 = generate_comparison_key("prompt1", "model_a", "model_b")
        key2 = generate_comparison_key("prompt1", "model_a", "model_b")
        assert key1 == key2

    def test_order_independent(self) -> None:
        """Test key is same regardless of model order."""
        key1 = generate_comparison_key("prompt1", "model_a", "model_b")
        key2 = generate_comparison_key("prompt1", "model_b", "model_a")
        assert key1 == key2

    def test_different_prompts_different_keys(self) -> None:
        """Test different prompts produce different keys."""
        key1 = generate_comparison_key("prompt1", "model_a", "model_b")
        key2 = generate_comparison_key("prompt2", "model_a", "model_b")
        assert key1 != key2

    def test_key_length(self) -> None:
        """Test key is 16 characters."""
        key = generate_comparison_key("prompt1", "model_a", "model_b")
        assert len(key) == 16


class TestGenerateRunId:
    """Tests for run ID generation."""

    def test_format(self) -> None:
        """Test run ID format."""
        run_id = generate_run_id()
        assert run_id.startswith("eval-")
        parts = run_id.split("-")
        assert len(parts) == 4  # eval, date, time, suffix

    def test_unique(self) -> None:
        """Test run IDs are unique."""
        # Note: This test may occasionally fail due to timing
        # but should pass in practice
        ids = {generate_run_id() for _ in range(10)}
        assert len(ids) >= 9  # Allow for rare collision


class TestComparisonResult:
    """Tests for ComparisonResult dataclass."""

    def test_to_dict(self) -> None:
        """Test conversion to dict."""
        result = ComparisonResult(
            comparison_id="abc123",
            prompt_id="prompt1",
            model_a_id="model_a",
            model_b_id="model_b",
            winner="A",
            confidence="high",
            judge_id="claude-3",
            timestamp="2024-01-01T00:00:00",
            position_order="AB",
            cost_usd=0.01,
        )

        d = result.to_dict()
        assert d["comparison_id"] == "abc123"
        assert d["winner"] == "A"

    def test_from_dict(self) -> None:
        """Test creation from dict."""
        d = {
            "comparison_id": "abc123",
            "prompt_id": "prompt1",
            "model_a_id": "model_a",
            "model_b_id": "model_b",
            "winner": "A",
            "confidence": "high",
            "judge_id": "claude-3",
            "timestamp": "2024-01-01T00:00:00",
            "position_order": "AB",
            "cost_usd": 0.01,
        }

        result = ComparisonResult.from_dict(d)
        assert result.comparison_id == "abc123"
        assert result.winner == "A"


class TestCheckpointData:
    """Tests for CheckpointData class."""

    def test_progress_percent(self) -> None:
        """Test progress percentage calculation."""
        data = CheckpointData(
            run_id="test",
            timestamp="2024-01-01T00:00:00",
            benchmark_version="lite",
            dataset_hash="abc123",
            judge_id="claude-3",
            completed_comparisons=50,
            total_comparisons=100,
            pending_pairs=[],
            current_elo_ratings={},
            comparison_results=[],
        )

        assert data.progress_percent == 50.0

    def test_progress_percent_zero(self) -> None:
        """Test progress with zero total."""
        data = CheckpointData(
            run_id="test",
            timestamp="2024-01-01T00:00:00",
            benchmark_version="lite",
            dataset_hash="abc123",
            judge_id="claude-3",
            completed_comparisons=0,
            total_comparisons=0,
            pending_pairs=[],
            current_elo_ratings={},
            comparison_results=[],
        )

        assert data.progress_percent == 0.0


class TestCheckpointManager:
    """Tests for CheckpointManager class."""

    @pytest.fixture
    def manager(self, tmp_results_dir: Path) -> CheckpointManager:
        """Create checkpoint manager with temp directory."""
        return CheckpointManager(tmp_results_dir)

    @pytest.fixture
    def sample_checkpoint(self) -> CheckpointData:
        """Create sample checkpoint data."""
        return create_initial_checkpoint(
            run_id="test-run-001",
            benchmark_version="lite",
            dataset_hash="abc123",
            judge_id="claude-3",
            model_ids=["model_a", "model_b"],
            pending_pairs=[("model_a", "model_b")],
        )

    def test_save_and_load(
        self, manager: CheckpointManager, sample_checkpoint: CheckpointData
    ) -> None:
        """Test saving and loading checkpoint."""
        path = manager.save(sample_checkpoint)

        assert path.exists()

        loaded = manager.load(sample_checkpoint.run_id)

        assert loaded is not None
        assert loaded.run_id == sample_checkpoint.run_id
        assert loaded.benchmark_version == sample_checkpoint.benchmark_version

    def test_load_nonexistent(self, manager: CheckpointManager) -> None:
        """Test loading nonexistent checkpoint."""
        loaded = manager.load("nonexistent")
        assert loaded is None

    def test_save_round(
        self, manager: CheckpointManager, sample_checkpoint: CheckpointData
    ) -> None:
        """Test saving round checkpoint."""
        path = manager.save_round(sample_checkpoint, 1)

        assert path.exists()
        assert "round_001.json" in str(path)

    def test_save_final(
        self, manager: CheckpointManager, sample_checkpoint: CheckpointData
    ) -> None:
        """Test saving final results."""
        path = manager.save_final(sample_checkpoint)

        assert path.exists()
        assert path.name == "final.json"

    def test_find_latest(
        self, manager: CheckpointManager, sample_checkpoint: CheckpointData
    ) -> None:
        """Test finding latest checkpoint."""
        # Save a checkpoint
        manager.save(sample_checkpoint)

        latest = manager.find_latest()

        assert latest is not None
        assert latest.run_id == sample_checkpoint.run_id

    def test_get_completed_comparisons(
        self, manager: CheckpointManager
    ) -> None:
        """Test extracting completed comparison IDs."""
        checkpoint = CheckpointData(
            run_id="test",
            timestamp="2024-01-01T00:00:00",
            benchmark_version="lite",
            dataset_hash="abc123",
            judge_id="claude-3",
            completed_comparisons=2,
            total_comparisons=10,
            pending_pairs=[],
            current_elo_ratings={},
            comparison_results=[
                {"comparison_id": "id1", "winner": "A"},
                {"comparison_id": "id2", "winner": "B"},
            ],
        )

        completed = manager.get_completed_comparisons(checkpoint)

        assert completed == {"id1", "id2"}


class TestCreateInitialCheckpoint:
    """Tests for create_initial_checkpoint function."""

    def test_creates_valid_checkpoint(self) -> None:
        """Test initial checkpoint creation."""
        checkpoint = create_initial_checkpoint(
            run_id="test-001",
            benchmark_version="base",
            dataset_hash="def456",
            judge_id="gpt-4",
            model_ids=["model_a", "model_b", "model_c"],
            pending_pairs=[("model_a", "model_b"), ("model_a", "model_c")],
        )

        assert checkpoint.run_id == "test-001"
        assert checkpoint.benchmark_version == "base"
        assert checkpoint.completed_comparisons == 0
        assert checkpoint.total_comparisons == 2
        assert len(checkpoint.pending_pairs) == 2
        assert len(checkpoint.current_elo_ratings) == 3
        assert all(r == 1500.0 for r in checkpoint.current_elo_ratings.values())
