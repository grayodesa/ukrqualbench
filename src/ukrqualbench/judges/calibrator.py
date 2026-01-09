"""Judge calibrator for validating LLM judges against gold standards.

Ensures judges meet quality thresholds before use in evaluation:
- MC agreement: >85%
- GEC F1: >80%
- Russism detection F1: >85%
- False positive rate: <15%
- Pairwise consistency: >90%
- Final acceptance threshold: judge_score > 0.80
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from ukrqualbench.core.schemas import (
    CalibrationResultData,
    WinnerChoice,
)
from ukrqualbench.judges.base import JudgeConfig, ModelClient
from ukrqualbench.judges.pairwise import PairwiseJudge
from ukrqualbench.judges.prompts import (
    format_gec_prompt,
    format_mc_prompt,
    format_russism_prompt,
)

if TYPE_CHECKING:
    pass


# Calibration thresholds
MC_ACCURACY_THRESHOLD = 0.85
GEC_F1_THRESHOLD = 0.80
RUSSISM_F1_THRESHOLD = 0.85
FALSE_POSITIVE_THRESHOLD = 0.15  # Maximum acceptable
PAIRWISE_CONSISTENCY_THRESHOLD = 0.90
POSITION_BIAS_THRESHOLD = 0.15  # Maximum acceptable deviation from 50%
LENGTH_BIAS_THRESHOLD = 0.30  # Maximum correlation with length
FINAL_SCORE_THRESHOLD = 0.80


@dataclass
class CalibrationTask:
    """Single calibration task with ground truth."""

    id: str
    task_type: str
    input_data: dict[str, Any]
    expected_output: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CalibrationMetrics:
    """Intermediate calibration metrics."""

    mc_correct: int = 0
    mc_total: int = 0
    gec_tp: int = 0
    gec_fp: int = 0
    gec_fn: int = 0
    russism_tp: int = 0
    russism_fp: int = 0
    russism_fn: int = 0
    false_positives: int = 0
    false_positive_total: int = 0
    pairwise_consistent: int = 0
    pairwise_total: int = 0
    position_a_wins: int = 0
    position_total: int = 0
    length_correlations: list[float] = field(default_factory=list)

    @property
    def mc_accuracy(self) -> float:
        """Calculate MC accuracy."""
        return self.mc_correct / self.mc_total if self.mc_total > 0 else 0.0

    @property
    def gec_precision(self) -> float:
        """Calculate GEC precision."""
        total = self.gec_tp + self.gec_fp
        return self.gec_tp / total if total > 0 else 0.0

    @property
    def gec_recall(self) -> float:
        """Calculate GEC recall."""
        total = self.gec_tp + self.gec_fn
        return self.gec_tp / total if total > 0 else 0.0

    @property
    def gec_f1(self) -> float:
        """Calculate GEC F1 score."""
        p, r = self.gec_precision, self.gec_recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def russism_precision(self) -> float:
        """Calculate russism detection precision."""
        total = self.russism_tp + self.russism_fp
        return self.russism_tp / total if total > 0 else 0.0

    @property
    def russism_recall(self) -> float:
        """Calculate russism detection recall."""
        total = self.russism_tp + self.russism_fn
        return self.russism_tp / total if total > 0 else 0.0

    @property
    def russism_f1(self) -> float:
        """Calculate russism detection F1 score."""
        p, r = self.russism_precision, self.russism_recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def false_positive_rate(self) -> float:
        """Calculate false positive rate."""
        if self.false_positive_total == 0:
            return 0.0
        return self.false_positives / self.false_positive_total

    @property
    def pairwise_consistency(self) -> float:
        """Calculate pairwise consistency."""
        if self.pairwise_total == 0:
            return 0.0
        return self.pairwise_consistent / self.pairwise_total

    @property
    def position_bias(self) -> float:
        """Calculate position bias (deviation from 50%)."""
        if self.position_total == 0:
            return 0.0
        a_rate = self.position_a_wins / self.position_total
        return abs(a_rate - 0.5)

    @property
    def length_bias_correlation(self) -> float:
        """Calculate average length bias correlation."""
        if not self.length_correlations:
            return 0.0
        return sum(abs(c) for c in self.length_correlations) / len(self.length_correlations)


class JudgeCalibrator:
    """Calibrates and validates LLM judges against gold standards.

    Runs a series of tests on calibration data to ensure the judge
    meets quality thresholds before being used in evaluation.

    Example:
        >>> calibrator = JudgeCalibrator(model_client)
        >>> result = await calibrator.calibrate(calibration_data)
        >>> if result.passed:
        ...     print("Judge passed calibration!")
        ... else:
        ...     print(f"Failed: {result.failure_reasons}")
    """

    def __init__(
        self,
        model: ModelClient,
        config: JudgeConfig | None = None,
        thresholds: dict[str, float] | None = None,
    ) -> None:
        """Initialize calibrator.

        Args:
            model: Model client for the judge.
            config: Optional judge configuration.
            thresholds: Optional custom thresholds.
        """
        self._model = model
        self._config = config or JudgeConfig()
        self._thresholds = thresholds or {}
        self._metrics = CalibrationMetrics()

    def _get_threshold(self, name: str, default: float) -> float:
        """Get threshold value, preferring custom over default."""
        return self._thresholds.get(name, default)

    async def calibrate(
        self,
        calibration_data: list[CalibrationTask],
    ) -> CalibrationResultData:
        """Run full calibration suite.

        Args:
            calibration_data: List of calibration tasks.

        Returns:
            Calibration result with pass/fail and detailed metrics.
        """
        # Reset metrics
        self._metrics = CalibrationMetrics()

        # Group tasks by type
        tasks_by_type: dict[str, list[CalibrationTask]] = {}
        for task in calibration_data:
            if task.task_type not in tasks_by_type:
                tasks_by_type[task.task_type] = []
            tasks_by_type[task.task_type].append(task)

        # Run calibration for each task type
        if "multiple_choice" in tasks_by_type:
            await self._calibrate_mc(tasks_by_type["multiple_choice"])

        if "gec" in tasks_by_type:
            await self._calibrate_gec(tasks_by_type["gec"])

        if "russism" in tasks_by_type:
            await self._calibrate_russism(tasks_by_type["russism"])

        if "false_positive" in tasks_by_type:
            await self._calibrate_false_positive(tasks_by_type["false_positive"])

        if "pairwise" in tasks_by_type:
            await self._calibrate_pairwise(tasks_by_type["pairwise"])

        # Compute final score and pass/fail
        return self._compute_result()

    async def _calibrate_mc(self, tasks: list[CalibrationTask]) -> None:
        """Calibrate on multiple choice tasks."""

        for task in tasks:
            question = task.input_data.get("question", "")
            options = task.input_data.get("options", [])
            expected = task.expected_output

            system_prompt, user_prompt = format_mc_prompt(question, options)

            try:
                # Create temporary judge for MC evaluation
                response = await self._call_model(system_prompt, user_prompt)
                parsed = self._parse_json_response(response.text)

                answer = parsed.get("answer", "").upper().strip()
                self._metrics.mc_total += 1

                if answer == expected:
                    self._metrics.mc_correct += 1

            except Exception:
                self._metrics.mc_total += 1  # Count as wrong

    async def _calibrate_gec(self, tasks: list[CalibrationTask]) -> None:
        """Calibrate on GEC tasks."""
        for task in tasks:
            original = task.input_data.get("original", "")
            reference = task.input_data.get("reference", "")
            expected_errors = task.expected_output or []

            # Model would correct, but for calibration we check if judge
            # correctly identifies errors
            system_prompt, user_prompt = format_gec_prompt(
                original=original,
                corrected=original,  # Judge evaluates uncorrected
                reference=reference,
            )

            try:
                response = await self._call_model(system_prompt, user_prompt)
                parsed = self._parse_json_response(response.text)

                found_errors = set(parsed.get("correct_fixes", []))
                expected_set = set(expected_errors)

                # Calculate TP, FP, FN
                tp = len(found_errors & expected_set)
                fp = len(found_errors - expected_set)
                fn = len(expected_set - found_errors)

                self._metrics.gec_tp += tp
                self._metrics.gec_fp += fp
                self._metrics.gec_fn += fn

            except Exception:
                self._metrics.gec_fn += len(expected_errors)

    async def _calibrate_russism(self, tasks: list[CalibrationTask]) -> None:
        """Calibrate on russism detection tasks."""
        for task in tasks:
            text = task.input_data.get("text", "")
            expected_russisms = task.expected_output or []

            system_prompt, user_prompt = format_russism_prompt(text)

            try:
                response = await self._call_model(system_prompt, user_prompt)
                parsed = self._parse_json_response(response.text)

                found_russisms = [r.get("text", "").lower() for r in parsed.get("russisms", [])]
                found_set = set(found_russisms)
                expected_set = {r.lower() for r in expected_russisms}

                tp = len(found_set & expected_set)
                fp = len(found_set - expected_set)
                fn = len(expected_set - found_set)

                self._metrics.russism_tp += tp
                self._metrics.russism_fp += fp
                self._metrics.russism_fn += fn

            except Exception:
                self._metrics.russism_fn += len(expected_russisms)

    async def _calibrate_false_positive(
        self,
        tasks: list[CalibrationTask],
    ) -> None:
        """Calibrate on false positive tasks."""
        for task in tasks:
            text = task.input_data.get("text", "")
            should_be_clean = task.expected_output  # True if no errors expected

            system_prompt, user_prompt = format_russism_prompt(text)

            try:
                response = await self._call_model(system_prompt, user_prompt)
                parsed = self._parse_json_response(response.text)

                found_errors = len(parsed.get("russisms", []))
                self._metrics.false_positive_total += 1

                # If text should be clean but errors were found = false positive
                if should_be_clean and found_errors > 0:
                    self._metrics.false_positives += 1

            except Exception:
                self._metrics.false_positive_total += 1

    async def _calibrate_pairwise(self, tasks: list[CalibrationTask]) -> None:
        """Calibrate on pairwise comparison tasks."""
        judge = PairwiseJudge(self._model, self._config, shuffle_positions=False)

        for task in tasks:
            prompt = task.input_data.get("prompt", "")
            response_a = task.input_data.get("response_a", "")
            response_b = task.input_data.get("response_b", "")
            _expected_winner = task.expected_output  # Reserved for future accuracy check

            # Test consistency with multiple rounds
            verdicts: list[WinnerChoice] = []
            for _ in range(3):
                try:
                    verdict = await judge.judge_pairwise(prompt, response_a, response_b)
                    verdicts.append(verdict.winner)
                except Exception:
                    verdicts.append(WinnerChoice.TIE)

            # Check if verdicts are consistent
            self._metrics.pairwise_total += 1
            if len(set(verdicts)) == 1:
                self._metrics.pairwise_consistent += 1

            # Track position bias (A vs B preference)
            self._metrics.position_total += len(verdicts)
            self._metrics.position_a_wins += sum(1 for v in verdicts if v == WinnerChoice.A)

            # Track length bias
            len_a = len(response_a)
            len_b = len(response_b)
            if len_a != len_b:
                longer_won = sum(
                    1
                    for v in verdicts
                    if (v == WinnerChoice.A and len_a > len_b)
                    or (v == WinnerChoice.B and len_b > len_a)
                )
                correlation = (longer_won / len(verdicts)) - 0.5
                self._metrics.length_correlations.append(correlation * 2)

    async def _call_model(self, system_prompt: str, user_prompt: str) -> Any:
        """Call model for calibration."""
        return await self._model.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens,
            json_mode=self._config.json_mode,
        )

    def _parse_json_response(self, text: str) -> dict[str, Any]:
        """Parse JSON from model response."""
        import json
        import re

        # Try to extract JSON from markdown
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if json_match:
            text = json_match.group(1)

        json_obj_match = re.search(r"\{[\s\S]*\}", text)
        if json_obj_match:
            text = json_obj_match.group(0)

        result: dict[str, Any] = json.loads(text)
        return result

    def _compute_result(self) -> CalibrationResultData:
        """Compute final calibration result."""
        m = self._metrics

        # Calculate individual pass/fail
        failures: list[str] = []

        mc_threshold = self._get_threshold("mc_accuracy", MC_ACCURACY_THRESHOLD)
        if m.mc_accuracy < mc_threshold:
            failures.append(f"MC accuracy {m.mc_accuracy:.2f} < {mc_threshold:.2f}")

        gec_threshold = self._get_threshold("gec_f1", GEC_F1_THRESHOLD)
        if m.gec_f1 < gec_threshold:
            failures.append(f"GEC F1 {m.gec_f1:.2f} < {gec_threshold:.2f}")

        russism_threshold = self._get_threshold("russism_f1", RUSSISM_F1_THRESHOLD)
        if m.russism_f1 < russism_threshold:
            failures.append(f"Russism F1 {m.russism_f1:.2f} < {russism_threshold:.2f}")

        fp_threshold = self._get_threshold("false_positive", FALSE_POSITIVE_THRESHOLD)
        if m.false_positive_rate > fp_threshold:
            failures.append(f"False positive rate {m.false_positive_rate:.2f} > {fp_threshold:.2f}")

        pairwise_threshold = self._get_threshold(
            "pairwise_consistency", PAIRWISE_CONSISTENCY_THRESHOLD
        )
        if m.pairwise_consistency < pairwise_threshold:
            failures.append(
                f"Pairwise consistency {m.pairwise_consistency:.2f} < {pairwise_threshold:.2f}"
            )

        position_threshold = self._get_threshold("position_bias", POSITION_BIAS_THRESHOLD)
        if m.position_bias > position_threshold:
            failures.append(f"Position bias {m.position_bias:.2f} > {position_threshold:.2f}")

        length_threshold = self._get_threshold("length_bias", LENGTH_BIAS_THRESHOLD)
        if m.length_bias_correlation > length_threshold:
            failures.append(f"Length bias {m.length_bias_correlation:.2f} > {length_threshold:.2f}")

        # Calculate weighted final score
        weights = {
            "mc": 0.15,
            "gec": 0.20,
            "russism": 0.20,
            "fp": 0.15,
            "pairwise": 0.15,
            "position": 0.10,
            "length": 0.05,
        }

        # Normalize metrics to 0-1 (higher = better)
        scores = {
            "mc": m.mc_accuracy,
            "gec": m.gec_f1,
            "russism": m.russism_f1,
            "fp": 1.0 - m.false_positive_rate,  # Invert (lower FP = better)
            "pairwise": m.pairwise_consistency,
            "position": 1.0 - (m.position_bias / 0.5),  # Normalize bias
            "length": 1.0 - (m.length_bias_correlation / 1.0),  # Normalize
        }

        # Clamp all scores to [0, 1]
        scores = {k: max(0.0, min(1.0, v)) for k, v in scores.items()}

        final_score = sum(scores[k] * weights[k] for k in weights)

        final_threshold = self._get_threshold("final_score", FINAL_SCORE_THRESHOLD)
        passed = final_score >= final_threshold and len(failures) == 0

        return CalibrationResultData(
            judge_id=self._model.model_id,
            passed=passed,
            mc_accuracy=m.mc_accuracy,
            gec_f1=m.gec_f1,
            russism_f1=m.russism_f1,
            false_positive_rate=m.false_positive_rate,
            pairwise_consistency=m.pairwise_consistency,
            position_bias=m.position_bias,
            length_bias_correlation=m.length_bias_correlation,
            final_score=final_score,
            failure_reasons=failures,
            timestamp=datetime.now(),
        )


def create_calibrator(
    model: ModelClient,
    temperature: float = 0.0,
    thresholds: dict[str, float] | None = None,
) -> JudgeCalibrator:
    """Factory function to create a judge calibrator.

    Args:
        model: Model client.
        temperature: Sampling temperature.
        thresholds: Optional custom thresholds.

    Returns:
        Configured JudgeCalibrator instance.
    """
    config = JudgeConfig(
        temperature=temperature,
        json_mode=True,
    )
    return JudgeCalibrator(model, config, thresholds)
