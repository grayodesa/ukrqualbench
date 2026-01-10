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

from collections.abc import Callable
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

ProgressCallback = Callable[[int, int, str], None]


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
    # Position bias: tracks if judge gives same answer regardless of position
    position_swaps_consistent: int = 0
    position_swaps_total: int = 0
    # Length bias: tracks if judge incorrectly favors longer response
    length_bias_incorrect: int = 0  # Cases where longer won but shorter was expected
    length_bias_applicable: int = 0  # Cases where lengths differ and we have ground truth
    # Accuracy against expected winner
    pairwise_correct: int = 0

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
        """Calculate position bias as inconsistency rate when positions are swapped."""
        if self.position_swaps_total == 0:
            return 0.0
        return 1.0 - (self.position_swaps_consistent / self.position_swaps_total)

    @property
    def length_bias_correlation(self) -> float:
        """Calculate length bias as rate of incorrect longer-wins verdicts."""
        if self.length_bias_applicable == 0:
            return 0.0
        return self.length_bias_incorrect / self.length_bias_applicable

    @property
    def pairwise_accuracy(self) -> float:
        """Calculate accuracy against expected winners."""
        if self.pairwise_total == 0:
            return 0.0
        return self.pairwise_correct / self.pairwise_total


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
        on_progress: ProgressCallback | None = None,
    ) -> CalibrationResultData:
        """Run full calibration suite.

        Args:
            calibration_data: List of calibration tasks.
            on_progress: Optional callback(current, total, task_type) for progress updates.

        Returns:
            Calibration result with pass/fail and detailed metrics.
        """
        self._metrics = CalibrationMetrics()
        self._on_progress = on_progress
        self._progress_current = 0
        self._progress_total = len(calibration_data)

        tasks_by_type: dict[str, list[CalibrationTask]] = {}
        for task in calibration_data:
            if task.task_type not in tasks_by_type:
                tasks_by_type[task.task_type] = []
            tasks_by_type[task.task_type].append(task)

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

        return self._compute_result()

    def _report_progress(self, task_type: str) -> None:
        """Report progress after completing a task."""
        self._progress_current += 1
        if self._on_progress:
            self._on_progress(self._progress_current, self._progress_total, task_type)

    async def _calibrate_mc(self, tasks: list[CalibrationTask]) -> None:
        for task in tasks:
            question = task.input_data.get("question", "")
            options = task.input_data.get("options", [])
            expected = task.expected_output

            system_prompt, user_prompt = format_mc_prompt(question, options)

            try:
                response = await self._call_model(system_prompt, user_prompt)
                parsed = self._parse_json_response(response.text)

                answer = parsed.get("answer", "").upper().strip()
                self._metrics.mc_total += 1

                if answer == expected:
                    self._metrics.mc_correct += 1

            except Exception:
                self._metrics.mc_total += 1

            self._report_progress("multiple_choice")

    async def _calibrate_gec(self, tasks: list[CalibrationTask]) -> None:
        """Calibrate on GEC tasks.

        Tests whether the judge can correctly identify what corrections
        are needed between original (with errors) and reference (correct).
        """
        for task in tasks:
            original = task.input_data.get("original", "")
            reference = task.input_data.get("reference", "")
            expected_errors = task.expected_output or []

            # Ask judge to evaluate the difference between original and reference
            system_prompt, user_prompt = format_gec_prompt(
                original=original,
                corrected=reference,  # Reference is the "corrected" version
                reference=reference,
            )

            try:
                response = await self._call_model(system_prompt, user_prompt)
                parsed = self._parse_json_response(response.text)

                found_errors = parsed.get("correct_fixes", [])
                found_text = " ".join(found_errors).lower()

                expected_keys = self._extract_gec_keys(expected_errors)

                matched = 0
                for key in expected_keys:
                    first_word = key.split()[0] if key else ""
                    if first_word and first_word in found_text:
                        matched += 1

                tp = matched
                fn = len(expected_keys) - tp
                fp = max(0, len(found_errors) - tp)

                self._metrics.gec_tp += tp
                self._metrics.gec_fp += fp
                self._metrics.gec_fn += fn

            except Exception:
                self._metrics.gec_fn += len(expected_errors)

            self._report_progress("gec")

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

            self._report_progress("russism")

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

            self._report_progress("false_positive")

    async def _calibrate_pairwise(self, tasks: list[CalibrationTask]) -> None:
        """Calibrate on pairwise comparison tasks."""
        judge = PairwiseJudge(self._model, self._config, shuffle_positions=False)

        for task in tasks:
            prompt = task.input_data.get("prompt", "")
            response_a = task.input_data.get("response_a", "")
            response_b = task.input_data.get("response_b", "")
            expected_winner = task.expected_output

            verdicts_normal: list[WinnerChoice] = []
            verdicts_swapped: list[WinnerChoice] = []

            for _ in range(2):
                try:
                    v_normal = await judge.judge_pairwise(prompt, response_a, response_b)
                    verdicts_normal.append(v_normal.winner)
                except Exception:
                    verdicts_normal.append(WinnerChoice.TIE)

                try:
                    v_swapped = await judge.judge_pairwise(prompt, response_b, response_a)
                    verdicts_swapped.append(v_swapped.winner)
                except Exception:
                    verdicts_swapped.append(WinnerChoice.TIE)

            self._metrics.pairwise_total += 1
            all_verdicts = verdicts_normal + verdicts_swapped
            if len(set(verdicts_normal)) == 1 and len(set(verdicts_swapped)) == 1:
                self._metrics.pairwise_consistent += 1

            self._metrics.position_swaps_total += 1
            normal_majority = self._get_majority_verdict(verdicts_normal)
            swapped_majority = self._get_majority_verdict(verdicts_swapped)
            if normal_majority != WinnerChoice.TIE and swapped_majority != WinnerChoice.TIE:
                normal_picks_original_a = normal_majority == WinnerChoice.A
                swapped_picks_original_a = swapped_majority == WinnerChoice.B
                if normal_picks_original_a == swapped_picks_original_a:
                    self._metrics.position_swaps_consistent += 1

            majority_verdict = self._get_majority_verdict(all_verdicts)
            if (expected_winner == "A" and majority_verdict == WinnerChoice.A) or (
                expected_winner == "B" and majority_verdict == WinnerChoice.B
            ):
                self._metrics.pairwise_correct += 1

            len_a, len_b = len(response_a), len(response_b)
            if len_a != len_b and majority_verdict != WinnerChoice.TIE:
                self._metrics.length_bias_applicable += 1
                longer_is_a = len_a > len_b
                judge_picked_longer = (majority_verdict == WinnerChoice.A and longer_is_a) or (
                    majority_verdict == WinnerChoice.B and not longer_is_a
                )
                expected_is_longer = (expected_winner == "A" and longer_is_a) or (
                    expected_winner == "B" and not longer_is_a
                )
                if judge_picked_longer and not expected_is_longer:
                    self._metrics.length_bias_incorrect += 1

            self._report_progress("pairwise")

    def _get_majority_verdict(self, verdicts: list[WinnerChoice]) -> WinnerChoice:
        """Get majority verdict from list, returns TIE if no clear majority."""
        if not verdicts:
            return WinnerChoice.TIE
        from collections import Counter

        counts = Counter(verdicts)
        most_common = counts.most_common(1)[0]
        if most_common[1] > len(verdicts) / 2:
            return most_common[0]
        return WinnerChoice.TIE

    async def _call_model(self, system_prompt: str, user_prompt: str) -> Any:
        """Call model for calibration."""
        return await self._model.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens,
            json_mode=self._config.json_mode,
        )

    def _extract_gec_keys(self, items: list[str]) -> set[str]:
        """Extract key error words from GEC correction descriptions."""
        import re

        keys: set[str] = set()
        for item in items:
            item_lower = item.lower()

            if "→" in item:
                keys.add(item.split("→")[0].strip().lower())
            elif "->" in item:
                keys.add(item.split("->")[0].strip().lower())
            else:
                quoted = re.findall(r"[«\"']([^»\"']+)[»\"']", item)
                if quoted:
                    for q in quoted:
                        keys.add(q.lower())
                elif " на " in item_lower:
                    parts = item_lower.split(" на ")
                    if parts[0]:
                        keys.add(parts[0].strip())

        return keys

    def _parse_json_response(self, text: str) -> dict[str, Any]:
        import json
        import re

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
