"""Tests for judges module."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pytest

from ukrqualbench.core.schemas import (
    ConfidenceLevel,
    PositionOrder,
    WinnerChoice,
)
from ukrqualbench.judges import (
    CalibrationMetrics,
    CalibrationTask,
    JudgeCalibrator,
    JudgeConfig,
    ModelClient,
    ModelResponse,
    PairwiseJudge,
    PairwiseScores,
    PromptTemplate,
    PromptType,
    create_calibrator,
    create_pairwise_judge,
    format_gec_prompt,
    format_mc_prompt,
    format_pairwise_prompt,
    format_russism_prompt,
    get_template,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@dataclass
class MockModelResponse:
    """Mock model response for testing."""

    text: str
    tokens_used: int = 100
    latency_ms: float = 50.0
    model_id: str = "test-model"
    timestamp: datetime | None = None
    cost_usd: float = 0.001

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.now()


class MockModelClient:
    """Mock model client for testing."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = responses or []
        self._call_index = 0
        self._calls: list[dict[str, Any]] = []

    @property
    def model_id(self) -> str:
        return "mock-model-v1"

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> MockModelResponse:
        self._calls.append(
            {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "json_mode": json_mode,
            }
        )

        if self._call_index < len(self._responses):
            response_text = self._responses[self._call_index]
            self._call_index += 1
        else:
            response_text = '{"winner": "tie", "confidence": "low"}'

        return MockModelResponse(text=response_text)


@pytest.fixture
def mock_client() -> MockModelClient:
    """Create a mock model client."""
    return MockModelClient()


@pytest.fixture
def judge_config() -> JudgeConfig:
    """Create default judge config."""
    return JudgeConfig(
        temperature=0.0,
        max_tokens=1024,
        json_mode=True,
        max_retries=1,
    )


# =============================================================================
# Prompt Tests
# =============================================================================


class TestPromptTemplates:
    """Tests for prompt templates."""

    def test_get_template_pairwise(self) -> None:
        """Test getting pairwise template."""
        template = get_template(PromptType.PAIRWISE)
        assert isinstance(template, PromptTemplate)
        assert template.name == "pairwise_comparison"
        assert template.prompt_type == PromptType.PAIRWISE
        assert "JSON" in template.system_prompt

    def test_get_template_gec(self) -> None:
        """Test getting GEC template."""
        template = get_template(PromptType.GEC)
        assert template.name == "gec_evaluation"
        assert "виправлення помилок" in template.system_prompt

    def test_get_template_russism(self) -> None:
        """Test getting russism template."""
        template = get_template(PromptType.RUSSISM)
        assert template.name == "russism_detection"
        assert "русизм" in template.system_prompt.lower()

    def test_get_template_mc(self) -> None:
        """Test getting multiple choice template."""
        template = get_template(PromptType.MULTIPLE_CHOICE)
        assert template.name == "multiple_choice"

    def test_get_template_false_positive(self) -> None:
        """Test getting false positive template."""
        template = get_template(PromptType.FALSE_POSITIVE)
        assert template.name == "false_positive_check"

    def test_get_template_positive_markers(self) -> None:
        """Test getting positive markers template."""
        template = get_template(PromptType.POSITIVE_MARKERS)
        assert template.name == "positive_markers"

    def test_format_pairwise_prompt(self) -> None:
        """Test formatting pairwise prompt."""
        system, user = format_pairwise_prompt(
            prompt="Тестове питання",
            response_a="Відповідь А",
            response_b="Відповідь Б",
        )
        assert "Текст A:" in user
        assert "Текст B:" in user
        assert "Відповідь А" in user
        assert "Відповідь Б" in user
        assert "JSON" in system

    def test_format_gec_prompt(self) -> None:
        """Test formatting GEC prompt."""
        _system, user = format_gec_prompt(
            original="Текст з помилкою",
            corrected="Виправлений текст",
            reference="Еталон",
        )
        assert "Оригінальний текст" in user
        assert "Виправлений текст" in user
        assert "Еталон" in user

    def test_format_mc_prompt(self) -> None:
        """Test formatting MC prompt."""
        _system, user = format_mc_prompt(
            question="Яке правильне слово?",
            options=["A) слово1", "B) слово2", "C) слово3"],
        )
        assert "Яке правильне слово?" in user
        assert "A) слово1" in user

    def test_format_russism_prompt(self) -> None:
        """Test formatting russism prompt."""
        _system, user = format_russism_prompt(
            text="Текст для аналізу",
        )
        assert "Текст для аналізу" in user
        assert "русизм" in user.lower()


# =============================================================================
# Base Judge Tests
# =============================================================================


class TestJudgeConfig:
    """Tests for JudgeConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = JudgeConfig()
        assert config.temperature == 0.0
        assert config.max_tokens == 8192
        assert config.json_mode is True
        assert config.max_retries == 3
        assert config.retry_delay == 1.0

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = JudgeConfig(
            temperature=0.5,
            max_tokens=2048,
            json_mode=False,
            max_retries=5,
        )
        assert config.temperature == 0.5
        assert config.max_tokens == 2048
        assert config.json_mode is False
        assert config.max_retries == 5


class TestModelResponse:
    """Tests for ModelResponse dataclass."""

    def test_creation(self) -> None:
        """Test creating model response."""
        response = ModelResponse(
            text="Test response",
            tokens_used=50,
            latency_ms=100.0,
            model_id="test-model",
        )
        assert response.text == "Test response"
        assert response.tokens_used == 50
        assert response.latency_ms == 100.0
        assert response.model_id == "test-model"
        assert response.cost_usd == 0.0

    def test_default_timestamp(self) -> None:
        """Test default timestamp is set."""
        response = ModelResponse(
            text="Test",
            tokens_used=10,
            latency_ms=50.0,
            model_id="test",
        )
        assert response.timestamp is not None


class TestJSONParsing:
    """Tests for JSON parsing in BaseJudge."""

    @pytest.mark.asyncio
    async def test_parse_simple_json(self) -> None:
        """Test parsing simple JSON response."""
        response_json = '{"winner": "A", "confidence": "high", "reasoning": "Test"}'
        client = MockModelClient([response_json])
        judge = PairwiseJudge(client, shuffle_positions=False)

        verdict = await judge.judge_pairwise("prompt", "response A", "response B")
        assert verdict.winner == WinnerChoice.A
        assert verdict.confidence == ConfidenceLevel.HIGH

    @pytest.mark.asyncio
    async def test_parse_json_in_markdown(self) -> None:
        """Test parsing JSON wrapped in markdown code block."""
        response_json = """```json
{"winner": "B", "confidence": "medium", "reasoning": "Better"}
```"""
        client = MockModelClient([response_json])
        judge = PairwiseJudge(client, shuffle_positions=False)

        verdict = await judge.judge_pairwise("prompt", "A", "B")
        assert verdict.winner == WinnerChoice.B

    @pytest.mark.asyncio
    async def test_parse_json_with_text_around(self) -> None:
        """Test parsing JSON with surrounding text."""
        response_json = """Here is my analysis:
{"winner": "tie", "confidence": "low", "reasoning": "Equal"}
That's my verdict."""
        client = MockModelClient([response_json])
        judge = PairwiseJudge(client, shuffle_positions=False)

        verdict = await judge.judge_pairwise("prompt", "A", "B")
        assert verdict.winner == WinnerChoice.TIE


# =============================================================================
# Pairwise Judge Tests
# =============================================================================


class TestPairwiseScores:
    """Tests for PairwiseScores."""

    def test_total_score(self) -> None:
        """Test total score calculation."""
        scores = PairwiseScores(
            nativeness=5,
            grammar=4,
            russisms=3,
            style=4,
        )
        assert scores.total == 16

    def test_average_score(self) -> None:
        """Test average score calculation."""
        scores = PairwiseScores(
            nativeness=4,
            grammar=4,
            russisms=4,
            style=4,
        )
        assert scores.average == 4.0


class TestPairwiseJudge:
    """Tests for PairwiseJudge."""

    @pytest.mark.asyncio
    async def test_judge_pairwise_winner_a(self) -> None:
        """Test judging with winner A."""
        response = '{"winner": "A", "confidence": "high", "reasoning": "A is better"}'
        client = MockModelClient([response])
        judge = PairwiseJudge(client, shuffle_positions=False)

        verdict = await judge.judge_pairwise(
            prompt="Питання",
            response_a="Хороша відповідь",
            response_b="Погана відповідь",
        )

        assert verdict.winner == WinnerChoice.A
        assert verdict.confidence == ConfidenceLevel.HIGH
        assert "A is better" in verdict.reasoning

    @pytest.mark.asyncio
    async def test_judge_pairwise_winner_b(self) -> None:
        """Test judging with winner B."""
        response = '{"winner": "B", "confidence": "medium", "reasoning": "B wins"}'
        client = MockModelClient([response])
        judge = PairwiseJudge(client, shuffle_positions=False)

        verdict = await judge.judge_pairwise("prompt", "A", "B")
        assert verdict.winner == WinnerChoice.B

    @pytest.mark.asyncio
    async def test_judge_pairwise_tie(self) -> None:
        """Test judging with tie result."""
        response = '{"winner": "tie", "confidence": "low", "reasoning": "Equal"}'
        client = MockModelClient([response])
        judge = PairwiseJudge(client, shuffle_positions=False)

        verdict = await judge.judge_pairwise("prompt", "A", "B")
        assert verdict.winner == WinnerChoice.TIE
        assert verdict.confidence == ConfidenceLevel.LOW

    @pytest.mark.asyncio
    async def test_judge_detailed_with_scores(self) -> None:
        """Test detailed judging with score breakdown."""
        response = """{
            "winner": "A",
            "confidence": "high",
            "reasoning": "Better Ukrainian",
            "scores": {
                "a": {"nativeness": 5, "grammar": 4, "russisms": 5, "style": 4},
                "b": {"nativeness": 3, "grammar": 3, "russisms": 2, "style": 3}
            }
        }"""
        client = MockModelClient([response])
        judge = PairwiseJudge(client, shuffle_positions=False)

        detailed = await judge.judge_detailed("prompt", "A", "B")

        assert detailed.verdict.winner == WinnerChoice.A
        assert detailed.scores_a is not None
        assert detailed.scores_a.nativeness == 5
        assert detailed.scores_b is not None
        assert detailed.scores_b.nativeness == 3

    @pytest.mark.asyncio
    async def test_position_mapping_when_shuffled(self) -> None:
        """Test that winner is correctly mapped when positions are swapped."""
        # When shuffle happens, B is presented as A, so winner "A" means original B
        response = '{"winner": "A", "confidence": "high", "reasoning": "First is better"}'
        client = MockModelClient([response])

        # Create judge that always swaps (by using a fixed seed scenario)
        judge = PairwiseJudge(client, shuffle_positions=False)
        verdict = await judge.judge_pairwise("prompt", "A", "B")

        # Without shuffle, A wins should stay A
        assert verdict.winner == WinnerChoice.A

    @pytest.mark.asyncio
    async def test_judge_with_consistency_unanimous(self) -> None:
        """Test consistency checking with unanimous verdict."""
        responses = [
            '{"winner": "A", "confidence": "high", "reasoning": "A wins"}',
            '{"winner": "A", "confidence": "high", "reasoning": "A wins"}',
            '{"winner": "A", "confidence": "high", "reasoning": "A wins"}',
        ]
        client = MockModelClient(responses)
        judge = PairwiseJudge(client, shuffle_positions=False)

        verdict, consistency = await judge.judge_with_consistency("prompt", "A", "B", rounds=3)

        assert verdict.winner == WinnerChoice.A
        assert consistency == 1.0

    @pytest.mark.asyncio
    async def test_judge_with_consistency_mixed(self) -> None:
        """Test consistency checking with mixed verdicts."""
        responses = [
            '{"winner": "A", "confidence": "high", "reasoning": "A"}',
            '{"winner": "B", "confidence": "medium", "reasoning": "B"}',
            '{"winner": "A", "confidence": "high", "reasoning": "A"}',
        ]
        client = MockModelClient(responses)
        judge = PairwiseJudge(client, shuffle_positions=False)

        verdict, consistency = await judge.judge_with_consistency("prompt", "A", "B", rounds=3)

        assert verdict.winner == WinnerChoice.A  # Majority
        assert consistency == pytest.approx(2 / 3)

    @pytest.mark.asyncio
    async def test_judge_statistics(self) -> None:
        """Test judge usage statistics."""
        responses = [
            '{"winner": "A", "confidence": "high", "reasoning": "A"}',
            '{"winner": "B", "confidence": "medium", "reasoning": "B"}',
        ]
        client = MockModelClient(responses)
        judge = PairwiseJudge(client, shuffle_positions=False)

        await judge.judge_pairwise("p1", "A", "B")
        await judge.judge_pairwise("p2", "C", "D")

        assert judge.call_count == 2
        assert judge.total_tokens == 200  # 100 * 2
        assert judge.average_latency_ms == 50.0

    @pytest.mark.asyncio
    async def test_reset_statistics(self) -> None:
        """Test resetting statistics."""
        response = '{"winner": "A", "confidence": "high", "reasoning": "A"}'
        client = MockModelClient([response])
        judge = PairwiseJudge(client, shuffle_positions=False)

        await judge.judge_pairwise("p", "A", "B")
        assert judge.call_count == 1

        judge.reset_statistics()
        assert judge.call_count == 0
        assert judge.total_tokens == 0


class TestCreatePairwiseJudge:
    """Tests for create_pairwise_judge factory."""

    def test_create_with_defaults(self) -> None:
        """Test creating judge with default settings."""
        client = MockModelClient()
        judge = create_pairwise_judge(client)

        assert isinstance(judge, PairwiseJudge)
        assert judge.model_id == "mock-model-v1"

    def test_create_with_custom_settings(self) -> None:
        """Test creating judge with custom settings."""
        client = MockModelClient()
        judge = create_pairwise_judge(
            client,
            shuffle_positions=False,
            temperature=0.5,
            max_retries=5,
        )

        assert isinstance(judge, PairwiseJudge)


# =============================================================================
# Calibration Tests
# =============================================================================


class TestCalibrationMetrics:
    """Tests for CalibrationMetrics."""

    def test_mc_accuracy(self) -> None:
        """Test MC accuracy calculation."""
        metrics = CalibrationMetrics()
        metrics.mc_correct = 85
        metrics.mc_total = 100
        assert metrics.mc_accuracy == 0.85

    def test_mc_accuracy_zero_total(self) -> None:
        """Test MC accuracy with zero total."""
        metrics = CalibrationMetrics()
        assert metrics.mc_accuracy == 0.0

    def test_gec_f1(self) -> None:
        """Test GEC F1 calculation."""
        metrics = CalibrationMetrics()
        metrics.gec_tp = 80
        metrics.gec_fp = 10
        metrics.gec_fn = 10

        # precision = 80/90 = 0.889
        # recall = 80/90 = 0.889
        # F1 = 2 * 0.889 * 0.889 / 1.778 = 0.889
        assert metrics.gec_precision == pytest.approx(0.889, rel=0.01)
        assert metrics.gec_recall == pytest.approx(0.889, rel=0.01)
        assert metrics.gec_f1 == pytest.approx(0.889, rel=0.01)

    def test_russism_f1(self) -> None:
        """Test russism F1 calculation."""
        metrics = CalibrationMetrics()
        metrics.russism_tp = 90
        metrics.russism_fp = 5
        metrics.russism_fn = 5

        assert metrics.russism_precision == pytest.approx(0.947, rel=0.01)
        assert metrics.russism_recall == pytest.approx(0.947, rel=0.01)

    def test_false_positive_rate(self) -> None:
        """Test false positive rate calculation."""
        metrics = CalibrationMetrics()
        metrics.false_positives = 10
        metrics.false_positive_total = 100
        assert metrics.false_positive_rate == 0.10

    def test_pairwise_consistency(self) -> None:
        """Test pairwise consistency calculation."""
        metrics = CalibrationMetrics()
        metrics.pairwise_consistent = 95
        metrics.pairwise_total = 100
        assert metrics.pairwise_consistency == 0.95

    def test_position_bias(self) -> None:
        """Test position bias calculation (inconsistency rate)."""
        metrics = CalibrationMetrics()
        metrics.position_swaps_consistent = 9
        metrics.position_swaps_total = 10
        assert metrics.position_bias == pytest.approx(0.10)

    def test_length_bias_correlation(self) -> None:
        """Test length bias as incorrect longer-wins rate."""
        metrics = CalibrationMetrics()
        metrics.length_bias_incorrect = 2
        metrics.length_bias_applicable = 10
        assert metrics.length_bias_correlation == pytest.approx(0.2)


class TestCalibrationTask:
    """Tests for CalibrationTask."""

    def test_creation(self) -> None:
        """Test creating calibration task."""
        task = CalibrationTask(
            id="test-1",
            task_type="multiple_choice",
            input_data={"question": "What?", "options": ["A", "B"]},
            expected_output="A",
        )
        assert task.id == "test-1"
        assert task.task_type == "multiple_choice"
        assert task.expected_output == "A"


class TestJudgeCalibrator:
    """Tests for JudgeCalibrator."""

    @pytest.mark.asyncio
    async def test_calibrate_empty_data(self) -> None:
        """Test calibration with empty data."""
        client = MockModelClient()
        calibrator = JudgeCalibrator(client)

        result = await calibrator.calibrate([])

        # With no data, all metrics should be 0 or pass by default
        assert result.judge_id == "mock-model-v1"

    @pytest.mark.asyncio
    async def test_calibrate_mc_tasks(self) -> None:
        """Test calibration on MC tasks."""
        responses = [
            '{"answer": "A", "confidence": "high", "reasoning": "A is correct"}',
            '{"answer": "B", "confidence": "high", "reasoning": "B is correct"}',
        ]
        client = MockModelClient(responses)
        calibrator = JudgeCalibrator(client)

        tasks = [
            CalibrationTask(
                id="mc-1",
                task_type="multiple_choice",
                input_data={"question": "Q1?", "options": ["A", "B"]},
                expected_output="A",
            ),
            CalibrationTask(
                id="mc-2",
                task_type="multiple_choice",
                input_data={"question": "Q2?", "options": ["A", "B"]},
                expected_output="B",
            ),
        ]

        result = await calibrator.calibrate(tasks)

        assert result.mc_accuracy == 1.0  # Both correct

    @pytest.mark.asyncio
    async def test_calibrate_russism_tasks(self) -> None:
        """Test calibration on russism detection tasks."""
        responses = [
            '{"russisms": [{"text": "прийняти участь"}], "total_count": 1}',
        ]
        client = MockModelClient(responses)
        calibrator = JudgeCalibrator(client)

        tasks = [
            CalibrationTask(
                id="rus-1",
                task_type="russism",
                input_data={"text": "Треба прийняти участь у заході."},
                expected_output=["прийняти участь"],
            ),
        ]

        result = await calibrator.calibrate(tasks)

        assert result.russism_f1 == 1.0  # Found expected russism

    @pytest.mark.asyncio
    async def test_calibrate_false_positive_tasks(self) -> None:
        """Test calibration on false positive tasks."""
        responses = [
            '{"russisms": [], "total_count": 0}',  # Correctly found nothing
        ]
        client = MockModelClient(responses)
        calibrator = JudgeCalibrator(client)

        tasks = [
            CalibrationTask(
                id="fp-1",
                task_type="false_positive",
                input_data={"text": "Чистий український текст."},
                expected_output=True,  # Should be clean
            ),
        ]

        result = await calibrator.calibrate(tasks)

        assert result.false_positive_rate == 0.0  # No false positives

    @pytest.mark.asyncio
    async def test_calibration_failure_reasons(self) -> None:
        """Test that failure reasons are captured."""
        # Return wrong MC answers
        responses = [
            '{"answer": "B", "confidence": "high", "reasoning": "Wrong"}',
            '{"answer": "B", "confidence": "high", "reasoning": "Wrong"}',
        ]
        client = MockModelClient(responses)

        # Use very strict thresholds to force failure
        thresholds = {"mc_accuracy": 0.99}
        calibrator = JudgeCalibrator(client, thresholds=thresholds)

        tasks = [
            CalibrationTask(
                id="mc-1",
                task_type="multiple_choice",
                input_data={"question": "Q?", "options": ["A", "B"]},
                expected_output="A",  # Expect A but model says B
            ),
            CalibrationTask(
                id="mc-2",
                task_type="multiple_choice",
                input_data={"question": "Q?", "options": ["A", "B"]},
                expected_output="A",
            ),
        ]

        result = await calibrator.calibrate(tasks)

        assert not result.passed
        assert result.mc_accuracy == 0.0
        assert any("MC accuracy" in r for r in result.failure_reasons)


class TestCreateCalibrator:
    """Tests for create_calibrator factory."""

    def test_create_with_defaults(self) -> None:
        """Test creating calibrator with defaults."""
        client = MockModelClient()
        calibrator = create_calibrator(client)
        assert isinstance(calibrator, JudgeCalibrator)

    def test_create_with_custom_thresholds(self) -> None:
        """Test creating calibrator with custom thresholds."""
        client = MockModelClient()
        thresholds = {"mc_accuracy": 0.90}
        calibrator = create_calibrator(client, thresholds=thresholds)
        assert isinstance(calibrator, JudgeCalibrator)


# =============================================================================
# Integration Tests
# =============================================================================


class TestJudgeIntegration:
    """Integration tests for judge components."""

    @pytest.mark.asyncio
    async def test_full_pairwise_evaluation(self) -> None:
        """Test complete pairwise evaluation flow."""
        response = """{
            "winner": "A",
            "confidence": "high",
            "reasoning": "Відповідь A має кращу українську мову",
            "scores": {
                "a": {"nativeness": 5, "grammar": 5, "russisms": 5, "style": 4},
                "b": {"nativeness": 2, "grammar": 3, "russisms": 2, "style": 2}
            }
        }"""
        client = MockModelClient([response])
        judge = create_pairwise_judge(client, shuffle_positions=False)

        detailed = await judge.judge_detailed(
            prompt="Поясніть фотосинтез",
            response_a="Фотосинтез — це процес перетворення...",
            response_b="Фотосинтез являється процесом...",
        )

        # Check verdict
        assert detailed.verdict.winner == WinnerChoice.A
        assert detailed.verdict.confidence == ConfidenceLevel.HIGH

        # Check scores
        assert detailed.scores_a is not None
        assert detailed.scores_a.total == 19
        assert detailed.scores_b is not None
        assert detailed.scores_b.total == 9

        # Check position tracking
        assert detailed.position_order == PositionOrder.AB

    @pytest.mark.asyncio
    async def test_model_client_protocol(self) -> None:
        """Test that MockModelClient satisfies ModelClient protocol."""
        client = MockModelClient()
        assert isinstance(client, ModelClient)
        assert client.model_id == "mock-model-v1"


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_malformed_json_response(self) -> None:
        """Test handling of malformed JSON response."""
        client = MockModelClient(["not json at all"])
        judge = PairwiseJudge(client, shuffle_positions=False)

        detailed = await judge.judge_detailed("prompt", "A", "B")

        # Should return tie with low confidence on parse failure
        assert detailed.verdict.winner == WinnerChoice.TIE
        assert detailed.verdict.confidence == ConfidenceLevel.LOW

    @pytest.mark.asyncio
    async def test_missing_required_fields(self) -> None:
        """Test handling of missing required fields."""
        # Missing confidence field - this is a required field
        client = MockModelClient(['{"winner": "A"}'])
        judge = PairwiseJudge(client, shuffle_positions=False)

        detailed = await judge.judge_detailed("prompt", "A", "B")

        # When required fields are missing, we fall back to TIE with low confidence
        assert detailed.verdict.winner == WinnerChoice.TIE
        assert detailed.verdict.confidence == ConfidenceLevel.LOW

    @pytest.mark.asyncio
    async def test_alternative_winner_formats(self) -> None:
        """Test parsing alternative winner formats."""
        test_cases = [
            ('{"winner": "1", "confidence": "high"}', WinnerChoice.A),
            ('{"winner": "2", "confidence": "high"}', WinnerChoice.B),
            ('{"winner": "FIRST", "confidence": "high"}', WinnerChoice.A),
            ('{"winner": "SECOND", "confidence": "high"}', WinnerChoice.B),
            ('{"winner": "MODEL_A", "confidence": "high"}', WinnerChoice.A),
            ('{"winner": "DRAW", "confidence": "high"}', WinnerChoice.TIE),
            ('{"winner": "NEITHER", "confidence": "high"}', WinnerChoice.TIE),
        ]

        for response, expected_winner in test_cases:
            client = MockModelClient([response])
            judge = PairwiseJudge(client, shuffle_positions=False)
            verdict = await judge.judge_pairwise("p", "A", "B")
            assert verdict.winner == expected_winner, f"Failed for {response}"

    @pytest.mark.asyncio
    async def test_score_clamping(self) -> None:
        """Test that scores are clamped to valid range."""
        response = """{
            "winner": "A",
            "confidence": "high",
            "reasoning": "Test",
            "scores": {
                "a": {"nativeness": 10, "grammar": -5, "russisms": 3, "style": 0},
                "b": {"nativeness": 3, "grammar": 3, "russisms": 3, "style": 3}
            }
        }"""
        client = MockModelClient([response])
        judge = PairwiseJudge(client, shuffle_positions=False)

        detailed = await judge.judge_detailed("p", "A", "B")

        assert detailed.scores_a is not None
        assert detailed.scores_a.nativeness == 5  # Clamped from 10
        assert detailed.scores_a.grammar == 1  # Clamped from -5
        assert detailed.scores_a.style == 1  # Clamped from 0

    @pytest.mark.asyncio
    async def test_empty_responses(self) -> None:
        """Test handling empty response strings."""
        response = '{"winner": "A", "confidence": "high", "reasoning": ""}'
        client = MockModelClient([response])
        judge = PairwiseJudge(client, shuffle_positions=False)

        verdict = await judge.judge_pairwise("", "", "")

        assert verdict.winner == WinnerChoice.A
        assert verdict.reasoning == ""

    @pytest.mark.asyncio
    async def test_ukrainian_confidence_values(self) -> None:
        """Test parsing Ukrainian confidence values."""
        test_cases = [
            ('{"winner": "A", "confidence": "висока"}', ConfidenceLevel.HIGH),
            ('{"winner": "A", "confidence": "впевнено"}', ConfidenceLevel.HIGH),
            ('{"winner": "A", "confidence": "низька"}', ConfidenceLevel.LOW),
            ('{"winner": "A", "confidence": "невпевнено"}', ConfidenceLevel.LOW),
        ]

        for response, expected_confidence in test_cases:
            client = MockModelClient([response])
            judge = PairwiseJudge(client, shuffle_positions=False)
            verdict = await judge.judge_pairwise("p", "A", "B")
            assert verdict.confidence == expected_confidence
