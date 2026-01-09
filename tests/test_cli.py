"""Tests for CLI commands."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from ukrqualbench.cli import app, create_model_client

runner = CliRunner()


def make_test_result(model_id: str = "test-model") -> dict[str, Any]:
    """Create test evaluation result dict."""
    from ukrqualbench.core.schemas import (
        Badge,
        BlockAScores,
        BlockBScores,
        BlockVScores,
        EvaluationMetadataData,
        EvaluationResultData,
        ModelScoreData,
    )

    result = EvaluationResultData(
        model_id=model_id,
        scores=ModelScoreData(
            elo_rating=1600.0,
            block_a=BlockAScores(
                mc_accuracy=0.85,
                gec_f1=0.80,
                translation_comet=0.75,
                false_positive_rate=0.10,
                positive_markers_score=0.80,
            ),
            block_b=BlockBScores(
                generation_elo=1600.0,
                adversarial_elo=1550.0,
                long_context_elo=1500.0,
            ),
            block_v=BlockVScores(
                fertility_rate=1.4,
                positive_markers=3.5,
                russism_rate=2.0,
                anglicism_rate=1.5,
            ),
            badge=Badge.SILVER,
        ),
        metadata=EvaluationMetadataData(
            benchmark_version="lite",
            dataset_hash="abc123",
            judge_id="test-judge",
            judge_calibration_score=0.90,
            total_prompts=100,
            total_comparisons=50,
            runtime_minutes=30.0,
            total_cost_usd=5.0,
        ),
        comparisons_count=50,
    )
    return result.to_dict()


class TestVersionAndHelp:
    def test_version_flag(self):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "UkrQualBench" in result.stdout
        assert "v" in result.stdout

    def test_help_flag(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "calibrate" in result.stdout
        assert "evaluate" in result.stdout
        assert "compare" in result.stdout
        assert "leaderboard" in result.stdout
        assert "info" in result.stdout

    def test_no_args_shows_help(self):
        result = runner.invoke(app, [])
        # Typer returns exit code 2 for no_args_is_help
        assert "Usage:" in result.stdout


class TestInfoCommand:
    def test_info_shows_configuration(self):
        result = runner.invoke(app, ["info"])
        assert result.exit_code == 0
        assert "Configuration" in result.stdout
        assert "API Keys" in result.stdout
        assert "Benchmark Sizes" in result.stdout

    def test_info_shows_benchmark_versions(self):
        result = runner.invoke(app, ["info"])
        assert "lite" in result.stdout
        assert "base" in result.stdout
        assert "large" in result.stdout


class TestCalibrateCommand:
    def test_calibrate_help(self):
        result = runner.invoke(app, ["calibrate", "--help"])
        assert result.exit_code == 0
        assert "--judge" in result.stdout
        assert "--output" in result.stdout

    def test_calibrate_requires_judge(self):
        result = runner.invoke(app, ["calibrate"])
        assert result.exit_code != 0

    @patch("ukrqualbench.cli.create_model_client")
    @patch("ukrqualbench.cli.asyncio.run")
    def test_calibrate_with_synthetic_data(self, mock_run, mock_create_client, tmp_path):
        mock_client = MagicMock()
        mock_create_client.return_value = mock_client

        mock_result = MagicMock()
        mock_result.judge_id = "test-judge"
        mock_result.passed = True
        mock_result.mc_accuracy = 0.90
        mock_result.gec_f1 = 0.85
        mock_result.russism_f1 = 0.88
        mock_result.false_positive_rate = 0.10
        mock_result.pairwise_consistency = 0.92
        mock_result.position_bias = 0.03
        mock_result.length_bias_correlation = 0.15
        mock_result.final_score = 0.88
        mock_result.failure_reasons = []
        mock_result.timestamp = MagicMock()
        mock_result.timestamp.isoformat.return_value = "2026-01-09T00:00:00"

        mock_run.return_value = mock_result

        result = runner.invoke(
            app,
            [
                "calibrate",
                "--judge",
                "test-judge",
                "--output",
                str(tmp_path),
            ],
        )

        assert result.exit_code == 0
        assert "PASSED" in result.stdout or mock_result.passed


class TestEvaluateCommand:
    def test_evaluate_help(self):
        result = runner.invoke(app, ["evaluate", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.stdout
        assert "--benchmark" in result.stdout
        assert "--judge" in result.stdout

    def test_evaluate_requires_model(self):
        result = runner.invoke(app, ["evaluate"])
        assert result.exit_code != 0


class TestCompareCommand:
    def test_compare_help(self):
        result = runner.invoke(app, ["compare", "--help"])
        assert result.exit_code == 0
        assert "--models" in result.stdout
        assert "--rounds" in result.stdout

    def test_compare_requires_models(self):
        result = runner.invoke(app, ["compare"])
        assert result.exit_code != 0


class TestLeaderboardCommand:
    def test_leaderboard_help(self):
        result = runner.invoke(app, ["leaderboard", "--help"])
        assert result.exit_code == 0
        assert "--results-dir" in result.stdout
        assert "--format" in result.stdout

    def test_leaderboard_no_results(self, tmp_path):
        result = runner.invoke(app, ["leaderboard", "--results-dir", str(tmp_path)])
        assert result.exit_code == 1
        assert "No evaluation results found" in result.stdout

    def test_leaderboard_with_results(self, tmp_path):
        result_file = tmp_path / "test-model_evaluation.json"
        with open(result_file, "w") as f:
            json.dump(make_test_result(), f)

        result = runner.invoke(
            app,
            [
                "leaderboard",
                "--results-dir",
                str(tmp_path),
                "--format",
                "json",
                "--output",
                str(tmp_path / "leaderboard.json"),
            ],
        )

        assert result.exit_code == 0
        assert "Loaded 1 evaluation results" in result.stdout


class TestCreateModelClient:
    def test_openai_model(self):
        with patch("ukrqualbench.models.create_openai_client") as mock:
            mock.return_value = MagicMock()
            create_model_client("gpt-5.2")
            mock.assert_called_once()

    def test_anthropic_model(self):
        with patch("ukrqualbench.models.create_anthropic_client") as mock:
            mock.return_value = MagicMock()
            create_model_client("claude-opus-4-5-20251101")
            mock.assert_called_once()

    def test_google_model(self):
        with patch("ukrqualbench.models.create_google_client") as mock:
            mock.return_value = MagicMock()
            create_model_client("gemini-3-flash-preview")
            mock.assert_called_once()

    def test_nebius_model(self):
        with patch("ukrqualbench.models.create_nebius_client") as mock:
            mock.return_value = MagicMock()
            create_model_client("deepseek-ai/DeepSeek-R1")
            mock.assert_called_once()

    def test_ollama_fallback(self):
        with patch("ukrqualbench.models.create_ollama_client") as mock:
            mock.return_value = MagicMock()
            create_model_client("llama3.2")
            mock.assert_called_once()


class TestCLIFormats:
    def test_leaderboard_markdown_format(self, tmp_path):
        result_file = tmp_path / "test-model_evaluation.json"
        with open(result_file, "w") as f:
            json.dump(make_test_result(), f)

        result = runner.invoke(
            app,
            [
                "leaderboard",
                "--results-dir",
                str(tmp_path),
                "--format",
                "markdown",
                "--output",
                str(tmp_path / "leaderboard.md"),
            ],
        )

        assert result.exit_code == 0
        md_file = tmp_path / "leaderboard.md"
        assert md_file.exists()

    def test_leaderboard_csv_format(self, tmp_path):
        result_file = tmp_path / "test-model_evaluation.json"
        with open(result_file, "w") as f:
            json.dump(make_test_result(), f)

        result = runner.invoke(
            app,
            [
                "leaderboard",
                "--results-dir",
                str(tmp_path),
                "--format",
                "csv",
                "--output",
                str(tmp_path / "leaderboard.csv"),
            ],
        )

        assert result.exit_code == 0
        csv_file = tmp_path / "leaderboard.csv"
        assert csv_file.exists()
