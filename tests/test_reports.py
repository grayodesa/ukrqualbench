"""Tests for report generation module."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from ukrqualbench.core.schemas import (
    Badge,
    BlockAScores,
    BlockBScores,
    BlockVScores,
    CalibrationResultData,
    EvaluationMetadataData,
    EvaluationResultData,
    ModelScoreData,
)
from ukrqualbench.reports import (
    AnalysisGenerator,
    LeaderboardEntry,
    LeaderboardGenerator,
    assign_badge,
    create_leaderboard,
    generate_calibration_html,
    generate_leaderboard_html,
)
from ukrqualbench.reports.analysis import analyze_results, generate_full_report


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_block_a_scores() -> BlockAScores:
    """Provide sample Block A scores."""
    return BlockAScores(
        mc_accuracy=0.87,
        gec_f1=0.82,
        translation_comet=0.78,
        false_positive_rate=0.12,
        positive_markers_score=0.0,
    )


@pytest.fixture
def sample_block_b_scores() -> BlockBScores:
    """Provide sample Block B scores."""
    return BlockBScores(
        generation_elo=1580.0,
        adversarial_elo=1560.0,
        long_context_elo=1550.0,
    )


@pytest.fixture
def sample_block_v_scores() -> BlockVScores:
    """Provide sample Block V scores."""
    return BlockVScores(
        fertility_rate=1.45,
        positive_markers=4.2,
        russism_rate=2.1,
        anglicism_rate=1.8,
    )


@pytest.fixture
def sample_model_scores(
    sample_block_a_scores: BlockAScores,
    sample_block_b_scores: BlockBScores,
    sample_block_v_scores: BlockVScores,
) -> ModelScoreData:
    """Provide sample model scores."""
    return ModelScoreData(
        elo_rating=1575.0,
        block_a=sample_block_a_scores,
        block_b=sample_block_b_scores,
        block_v=sample_block_v_scores,
        badge=Badge.SILVER,
    )


@pytest.fixture
def sample_metadata() -> EvaluationMetadataData:
    """Provide sample evaluation metadata."""
    return EvaluationMetadataData(
        benchmark_version="base",
        dataset_hash="abc123",
        judge_id="claude-3.5-haiku",
        judge_calibration_score=0.92,
        total_prompts=550,
        total_comparisons=150,
        runtime_minutes=45.5,
        total_cost_usd=12.50,
    )


@pytest.fixture
def sample_evaluation_result(
    sample_model_scores: ModelScoreData,
    sample_metadata: EvaluationMetadataData,
) -> EvaluationResultData:
    """Provide sample evaluation result."""
    return EvaluationResultData(
        model_id="gpt-5.2",
        scores=sample_model_scores,
        metadata=sample_metadata,
        comparisons_count=150,
    )


@pytest.fixture
def multiple_evaluation_results() -> list[EvaluationResultData]:
    """Provide multiple evaluation results for testing."""
    results = []

    # Model 1: Gold quality
    results.append(
        EvaluationResultData(
            model_id="model-gold",
            scores=ModelScoreData(
                elo_rating=1680.0,
                block_a=BlockAScores(
                    mc_accuracy=0.92,
                    gec_f1=0.88,
                    translation_comet=0.85,
                    false_positive_rate=0.08,
                ),
                block_b=BlockBScores(
                    generation_elo=1650.0,
                    adversarial_elo=1640.0,
                    long_context_elo=1630.0,
                ),
                block_v=BlockVScores(
                    fertility_rate=1.35,
                    positive_markers=6.5,
                    russism_rate=0.8,
                    anglicism_rate=1.2,
                ),
                badge=Badge.GOLD,
            ),
            metadata=EvaluationMetadataData(
                benchmark_version="base",
                dataset_hash="abc123",
                judge_id="claude-3.5-haiku",
                judge_calibration_score=0.92,
                total_prompts=550,
                total_comparisons=150,
                runtime_minutes=45.5,
                total_cost_usd=12.50,
            ),
            comparisons_count=150,
        )
    )

    # Model 2: Silver quality
    results.append(
        EvaluationResultData(
            model_id="model-silver",
            scores=ModelScoreData(
                elo_rating=1580.0,
                block_a=BlockAScores(
                    mc_accuracy=0.85,
                    gec_f1=0.82,
                    translation_comet=0.78,
                    false_positive_rate=0.12,
                ),
                block_b=BlockBScores(
                    generation_elo=1560.0,
                    adversarial_elo=1550.0,
                    long_context_elo=1540.0,
                ),
                block_v=BlockVScores(
                    fertility_rate=1.55,
                    positive_markers=4.0,
                    russism_rate=2.5,
                    anglicism_rate=2.0,
                ),
                badge=Badge.SILVER,
            ),
            metadata=EvaluationMetadataData(
                benchmark_version="base",
                dataset_hash="abc123",
                judge_id="claude-3.5-haiku",
                judge_calibration_score=0.92,
                total_prompts=550,
                total_comparisons=140,
                runtime_minutes=42.0,
                total_cost_usd=11.00,
            ),
            comparisons_count=140,
        )
    )

    # Model 3: Bronze quality
    results.append(
        EvaluationResultData(
            model_id="model-bronze",
            scores=ModelScoreData(
                elo_rating=1480.0,
                block_a=BlockAScores(
                    mc_accuracy=0.78,
                    gec_f1=0.72,
                    translation_comet=0.70,
                    false_positive_rate=0.18,
                ),
                block_b=BlockBScores(
                    generation_elo=1470.0,
                    adversarial_elo=1465.0,
                    long_context_elo=1460.0,
                ),
                block_v=BlockVScores(
                    fertility_rate=1.85,
                    positive_markers=2.0,
                    russism_rate=4.5,
                    anglicism_rate=3.5,
                ),
                badge=Badge.BRONZE,
            ),
            metadata=EvaluationMetadataData(
                benchmark_version="base",
                dataset_hash="abc123",
                judge_id="claude-3.5-haiku",
                judge_calibration_score=0.92,
                total_prompts=550,
                total_comparisons=130,
                runtime_minutes=40.0,
                total_cost_usd=10.00,
            ),
            comparisons_count=130,
        )
    )

    return results


@pytest.fixture
def sample_calibration_result() -> CalibrationResultData:
    """Provide sample calibration result."""
    return CalibrationResultData(
        judge_id="claude-3.5-haiku",
        passed=True,
        mc_accuracy=0.89,
        gec_f1=0.84,
        russism_f1=0.87,
        false_positive_rate=0.10,
        pairwise_consistency=0.93,
        position_bias=0.05,
        length_bias_correlation=0.02,
        final_score=0.88,
        failure_reasons=[],
    )


# ============================================================================
# Badge Assignment Tests
# ============================================================================


class TestBadgeAssignment:
    """Tests for badge assignment logic."""

    def test_assign_badge_gold(self) -> None:
        """Test gold badge assignment."""
        scores = ModelScoreData(
            elo_rating=1680.0,
            block_a=BlockAScores(
                mc_accuracy=0.90, gec_f1=0.85, translation_comet=0.82,
                false_positive_rate=0.08,
            ),
            block_b=BlockBScores(
                generation_elo=1650.0, adversarial_elo=1640.0, long_context_elo=1630.0,
            ),
            block_v=BlockVScores(
                fertility_rate=1.4, positive_markers=6.0,
                russism_rate=0.8, anglicism_rate=1.0,
            ),
            badge=Badge.NONE,
        )

        badge = assign_badge(scores)
        assert badge == Badge.GOLD

    def test_assign_badge_silver(self) -> None:
        """Test silver badge assignment."""
        scores = ModelScoreData(
            elo_rating=1580.0,
            block_a=BlockAScores(
                mc_accuracy=0.85, gec_f1=0.82, translation_comet=0.78,
                false_positive_rate=0.12,
            ),
            block_b=BlockBScores(
                generation_elo=1560.0, adversarial_elo=1550.0, long_context_elo=1540.0,
            ),
            block_v=BlockVScores(
                fertility_rate=1.6, positive_markers=4.0,
                russism_rate=2.5, anglicism_rate=2.0,
            ),
            badge=Badge.NONE,
        )

        badge = assign_badge(scores)
        assert badge == Badge.SILVER

    def test_assign_badge_bronze(self) -> None:
        """Test bronze badge assignment."""
        scores = ModelScoreData(
            elo_rating=1480.0,
            block_a=BlockAScores(
                mc_accuracy=0.78, gec_f1=0.72, translation_comet=0.70,
                false_positive_rate=0.18,
            ),
            block_b=BlockBScores(
                generation_elo=1470.0, adversarial_elo=1465.0, long_context_elo=1460.0,
            ),
            block_v=BlockVScores(
                fertility_rate=1.9, positive_markers=1.5,
                russism_rate=4.5, anglicism_rate=3.5,
            ),
            badge=Badge.NONE,
        )

        badge = assign_badge(scores)
        assert badge == Badge.BRONZE

    def test_assign_badge_not_recommended(self) -> None:
        """Test not recommended badge assignment."""
        scores = ModelScoreData(
            elo_rating=1300.0,
            block_a=BlockAScores(
                mc_accuracy=0.60, gec_f1=0.55, translation_comet=0.50,
                false_positive_rate=0.30,
            ),
            block_b=BlockBScores(
                generation_elo=1320.0, adversarial_elo=1310.0, long_context_elo=1300.0,
            ),
            block_v=BlockVScores(
                fertility_rate=2.8, positive_markers=0.3,
                russism_rate=12.0, anglicism_rate=8.0,
            ),
            badge=Badge.NONE,
        )

        badge = assign_badge(scores)
        assert badge == Badge.NOT_RECOMMENDED


# ============================================================================
# Leaderboard Tests
# ============================================================================


class TestLeaderboardEntry:
    """Tests for LeaderboardEntry dataclass."""

    def test_to_dict_minimal(self) -> None:
        """Test minimal dict conversion."""
        entry = LeaderboardEntry(
            rank=1,
            model_id="test-model",
            elo_rating=1600.0,
            badge=Badge.GOLD,
        )

        result = entry.to_dict(include_details=False)

        assert result["rank"] == 1
        assert result["model_id"] == "test-model"
        assert result["elo_rating"] == 1600.0
        assert result["badge"] == "gold"
        assert "block_a" not in result

    def test_to_dict_detailed(self) -> None:
        """Test detailed dict conversion."""
        entry = LeaderboardEntry(
            rank=1,
            model_id="test-model",
            elo_rating=1600.0,
            badge=Badge.GOLD,
            mc_accuracy=0.90,
            russism_rate=0.8,
            wins=10,
            losses=2,
            ties=1,
        )

        result = entry.to_dict(include_details=True)

        assert result["rank"] == 1
        assert "block_a" in result
        assert result["block_a"]["mc_accuracy"] == 0.90
        assert "block_v" in result
        assert result["block_v"]["russism_rate"] == 0.8
        assert result["statistics"]["wins"] == 10


class TestLeaderboardGenerator:
    """Tests for LeaderboardGenerator class."""

    def test_add_result(self, sample_evaluation_result: EvaluationResultData) -> None:
        """Test adding evaluation result."""
        generator = LeaderboardGenerator()
        generator.add_result(sample_evaluation_result, wins=5, losses=2, ties=1)

        assert len(generator.entries) == 1
        assert generator.entries[0].model_id == "gpt-5.2"
        assert generator.entries[0].wins == 5

    def test_add_from_results(
        self, multiple_evaluation_results: list[EvaluationResultData]
    ) -> None:
        """Test adding multiple results."""
        generator = LeaderboardGenerator()
        generator.add_from_results(multiple_evaluation_results)

        assert len(generator.entries) == 3

    def test_finalize_sorts_by_elo(
        self, multiple_evaluation_results: list[EvaluationResultData]
    ) -> None:
        """Test finalize sorts entries by ELO."""
        generator = LeaderboardGenerator()
        generator.add_from_results(multiple_evaluation_results)
        generator.finalize()

        # Should be sorted by ELO descending
        assert generator.entries[0].model_id == "model-gold"
        assert generator.entries[1].model_id == "model-silver"
        assert generator.entries[2].model_id == "model-bronze"

    def test_finalize_assigns_ranks(
        self, multiple_evaluation_results: list[EvaluationResultData]
    ) -> None:
        """Test finalize assigns correct ranks."""
        generator = LeaderboardGenerator()
        generator.add_from_results(multiple_evaluation_results)
        generator.finalize()

        assert generator.entries[0].rank == 1
        assert generator.entries[1].rank == 2
        assert generator.entries[2].rank == 3

    def test_finalize_creates_metadata(
        self, multiple_evaluation_results: list[EvaluationResultData]
    ) -> None:
        """Test finalize creates metadata."""
        generator = LeaderboardGenerator()
        generator.add_from_results(multiple_evaluation_results)
        generator.finalize(benchmark_version="base", judge_id="test-judge", total_prompts=100)

        assert generator.metadata is not None
        assert generator.metadata.benchmark_version == "base"
        assert generator.metadata.judge_id == "test-judge"
        assert generator.metadata.total_models == 3

    def test_get_entries_with_limit(
        self, multiple_evaluation_results: list[EvaluationResultData]
    ) -> None:
        """Test getting entries with limit."""
        generator = LeaderboardGenerator()
        generator.add_from_results(multiple_evaluation_results)
        generator.finalize()

        entries = generator.get_entries(limit=2)
        assert len(entries) == 2

    def test_get_entries_with_badge_filter(
        self, multiple_evaluation_results: list[EvaluationResultData]
    ) -> None:
        """Test getting entries with badge filter."""
        generator = LeaderboardGenerator()
        generator.add_from_results(multiple_evaluation_results)
        generator.finalize()

        entries = generator.get_entries(badge_filter=Badge.GOLD)
        assert len(entries) == 1
        assert entries[0].badge == Badge.GOLD

    def test_to_json(
        self, multiple_evaluation_results: list[EvaluationResultData]
    ) -> None:
        """Test JSON export."""
        generator = LeaderboardGenerator()
        generator.add_from_results(multiple_evaluation_results)
        generator.finalize()

        json_str = generator.to_json()
        data = json.loads(json_str)

        assert "leaderboard" in data
        assert len(data["leaderboard"]) == 3
        assert "metadata" in data

    def test_to_csv(
        self, multiple_evaluation_results: list[EvaluationResultData]
    ) -> None:
        """Test CSV export."""
        generator = LeaderboardGenerator()
        generator.add_from_results(multiple_evaluation_results)
        generator.finalize()

        csv_str = generator.to_csv()

        assert "rank,model_id,elo_rating,badge" in csv_str
        assert "model-gold" in csv_str

    def test_to_table_markdown(
        self, multiple_evaluation_results: list[EvaluationResultData]
    ) -> None:
        """Test Markdown table export."""
        generator = LeaderboardGenerator()
        generator.add_from_results(multiple_evaluation_results)
        generator.finalize()

        table = generator.to_table(format="markdown")

        assert "|" in table
        assert "Rank" in table
        assert "Model Id" in table

    def test_to_table_unicode(
        self, multiple_evaluation_results: list[EvaluationResultData]
    ) -> None:
        """Test Unicode table export."""
        generator = LeaderboardGenerator()
        generator.add_from_results(multiple_evaluation_results)
        generator.finalize()

        table = generator.to_table(format="unicode")

        assert "\u2502" in table  # Unicode box drawing

    def test_to_table_ascii(
        self, multiple_evaluation_results: list[EvaluationResultData]
    ) -> None:
        """Test ASCII table export."""
        generator = LeaderboardGenerator()
        generator.add_from_results(multiple_evaluation_results)
        generator.finalize()

        table = generator.to_table(format="ascii")

        assert "+" in table
        assert "-" in table


class TestCreateLeaderboard:
    """Tests for create_leaderboard convenience function."""

    def test_create_leaderboard(
        self, multiple_evaluation_results: list[EvaluationResultData]
    ) -> None:
        """Test create_leaderboard function."""
        leaderboard = create_leaderboard(
            multiple_evaluation_results,
            benchmark_version="base",
            judge_id="test-judge",
        )

        assert len(leaderboard.entries) == 3
        assert leaderboard.entries[0].rank == 1
        assert leaderboard.metadata is not None


# ============================================================================
# HTML Report Tests
# ============================================================================


class TestHTMLReports:
    """Tests for HTML report generation."""

    def test_generate_leaderboard_html(
        self, multiple_evaluation_results: list[EvaluationResultData]
    ) -> None:
        """Test leaderboard HTML generation."""
        leaderboard = create_leaderboard(multiple_evaluation_results)
        html = generate_leaderboard_html(leaderboard)

        assert "<html" in html
        assert "UkrQualBench Leaderboard" in html
        assert "model-gold" in html
        assert "model-silver" in html

    def test_generate_leaderboard_html_saves_file(
        self,
        multiple_evaluation_results: list[EvaluationResultData],
        tmp_path: Path,
    ) -> None:
        """Test leaderboard HTML file saving."""
        leaderboard = create_leaderboard(multiple_evaluation_results)
        output_path = tmp_path / "leaderboard.html"

        html = generate_leaderboard_html(leaderboard, output_path=output_path)

        assert output_path.exists()
        assert output_path.read_text() == html

    def test_generate_calibration_html(
        self, sample_calibration_result: CalibrationResultData
    ) -> None:
        """Test calibration HTML generation."""
        html = generate_calibration_html(sample_calibration_result)

        assert "<html" in html
        assert "Calibration" in html
        assert "claude-3.5-haiku" in html
        assert "PASSED" in html

    def test_generate_calibration_html_failed(self) -> None:
        """Test calibration HTML for failed calibration."""
        result = CalibrationResultData(
            judge_id="bad-judge",
            passed=False,
            mc_accuracy=0.60,
            gec_f1=0.55,
            russism_f1=0.50,
            false_positive_rate=0.30,
            pairwise_consistency=0.70,
            position_bias=0.20,
            length_bias_correlation=0.15,
            final_score=0.55,
            failure_reasons=["MC accuracy below threshold", "GEC F1 below threshold"],
        )

        html = generate_calibration_html(result)

        assert "FAILED" in html
        assert "MC accuracy below threshold" in html


# ============================================================================
# Analysis Report Tests
# ============================================================================


class TestAnalysisGenerator:
    """Tests for AnalysisGenerator class."""

    def test_analyze_model(self, sample_evaluation_result: EvaluationResultData) -> None:
        """Test model analysis generation."""
        generator = AnalysisGenerator()
        analysis = generator.analyze_model(sample_evaluation_result, wins=5, losses=2, ties=1)

        assert analysis.model_id == "gpt-5.2"
        assert analysis.overall_rating in ["excellent", "good", "acceptable", "poor"]
        assert isinstance(analysis.strengths, list)
        assert isinstance(analysis.weaknesses, list)

    def test_analyze_model_strengths_detection(self) -> None:
        """Test strength detection in analysis."""
        result = EvaluationResultData(
            model_id="strong-model",
            scores=ModelScoreData(
                elo_rating=1650.0,
                block_a=BlockAScores(
                    mc_accuracy=0.95,  # Very high
                    gec_f1=0.90,  # Very high
                    translation_comet=0.85,
                    false_positive_rate=0.05,
                ),
                block_b=BlockBScores(
                    generation_elo=1600.0,
                    adversarial_elo=1580.0,
                    long_context_elo=1560.0,
                ),
                block_v=BlockVScores(
                    fertility_rate=1.3,  # Very low (good)
                    positive_markers=7.0,  # Very high
                    russism_rate=0.5,  # Very low (good)
                    anglicism_rate=1.0,
                ),
                badge=Badge.GOLD,
            ),
            metadata=EvaluationMetadataData(
                benchmark_version="base",
                dataset_hash="abc123",
                judge_id="test",
                judge_calibration_score=0.90,
                total_prompts=100,
                total_comparisons=50,
                runtime_minutes=30.0,
                total_cost_usd=5.0,
            ),
            comparisons_count=50,
        )

        generator = AnalysisGenerator()
        analysis = generator.analyze_model(result)

        assert any("accuracy" in s.lower() for s in analysis.strengths)
        assert any("russism" in s.lower() or "marker" in s.lower() for s in analysis.strengths)

    def test_generate_summary(
        self, multiple_evaluation_results: list[EvaluationResultData]
    ) -> None:
        """Test summary generation."""
        generator = AnalysisGenerator()
        summary = generator.generate_summary(
            multiple_evaluation_results,
            benchmark_version="base",
            judge_id="test-judge",
        )

        assert summary.total_models == 3
        assert summary.benchmark_version == "base"
        assert summary.gold_count == 1
        assert summary.silver_count == 1
        assert summary.bronze_count == 1

    def test_analyze_comparison(
        self, multiple_evaluation_results: list[EvaluationResultData]
    ) -> None:
        """Test head-to-head comparison analysis."""
        generator = AnalysisGenerator()
        comparison = generator.analyze_comparison(
            multiple_evaluation_results[0],  # Gold
            multiple_evaluation_results[2],  # Bronze
        )

        assert comparison.winner == "model-gold"
        assert comparison.margin > 0

    def test_to_json(
        self, multiple_evaluation_results: list[EvaluationResultData]
    ) -> None:
        """Test JSON export."""
        generator = AnalysisGenerator()
        for result in multiple_evaluation_results:
            generator.analyze_model(result)
        generator.generate_summary(multiple_evaluation_results)

        json_str = generator.to_json()
        data = json.loads(json_str)

        assert "models" in data
        assert "summary" in data
        assert len(data["models"]) == 3

    def test_save_report_json(
        self,
        multiple_evaluation_results: list[EvaluationResultData],
        tmp_path: Path,
    ) -> None:
        """Test JSON report saving."""
        generator = AnalysisGenerator()
        for result in multiple_evaluation_results:
            generator.analyze_model(result)

        output_path = tmp_path / "analysis.json"
        saved_path = generator.save_report(output_path, format="json")

        assert saved_path.exists()
        data = json.loads(saved_path.read_text())
        assert "models" in data

    def test_save_report_markdown(
        self,
        multiple_evaluation_results: list[EvaluationResultData],
        tmp_path: Path,
    ) -> None:
        """Test Markdown report saving."""
        generator = AnalysisGenerator()
        for result in multiple_evaluation_results:
            generator.analyze_model(result)
        generator.generate_summary(multiple_evaluation_results)

        output_path = tmp_path / "analysis.md"
        saved_path = generator.save_report(output_path, format="md")

        assert saved_path.exists()
        content = saved_path.read_text()
        assert "# UkrQualBench" in content
        assert "## Summary" in content


class TestAnalyzeResults:
    """Tests for analyze_results convenience function."""

    def test_analyze_results(
        self, multiple_evaluation_results: list[EvaluationResultData]
    ) -> None:
        """Test analyze_results function."""
        generator = analyze_results(
            multiple_evaluation_results,
            benchmark_version="base",
            judge_id="test-judge",
        )

        assert len(generator._model_analyses) == 3
        assert generator._summary is not None


class TestGenerateFullReport:
    """Tests for generate_full_report function."""

    def test_generate_full_report(
        self,
        multiple_evaluation_results: list[EvaluationResultData],
        tmp_path: Path,
    ) -> None:
        """Test full report generation."""
        outputs = generate_full_report(
            multiple_evaluation_results,
            output_dir=tmp_path,
            benchmark_version="base",
            judge_id="test-judge",
        )

        assert "json" in outputs
        assert "md" in outputs
        assert outputs["json"].exists()
        assert outputs["md"].exists()
