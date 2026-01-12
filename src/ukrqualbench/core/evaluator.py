"""Main evaluation engine for UkrQualBench.

Orchestrates the complete benchmark evaluation process:
- Loading benchmark tasks
- Coordinating model API calls
- Running pairwise comparisons via judge
- Managing Swiss-system tournament rounds
- Tracking progress with checkpoints
- Calculating final scores and generating results
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ukrqualbench.core.checkpoint import CheckpointManager
from ukrqualbench.core.circuit_breaker import CircuitBreaker
from ukrqualbench.core.config import Config
from ukrqualbench.core.metrics import MetricsCollector, get_metrics
from ukrqualbench.core.pairwise import PairingStrategy, PairwiseEngine
from ukrqualbench.core.schemas import (
    Badge,
    BlockAScores,
    BlockBScores,
    BlockVScores,
    EvaluationMetadataData,
    EvaluationResultData,
    ModelEvaluationData,
    ModelScoreData,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ukrqualbench.judges.base import BaseJudge
    from ukrqualbench.models.base import BaseModelClient

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkTask:
    """A single benchmark task."""

    id: str
    type: str  # "multiple_choice", "gec", "translation", "generation", etc.
    category: str
    prompt: str
    reference: str | None = None  # For tasks with reference answers
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationProgress:
    """Progress tracking for evaluation."""

    total_tasks: int = 0
    completed_tasks: int = 0
    total_comparisons: int = 0
    completed_comparisons: int = 0
    current_round: int = 0
    total_rounds: int = 0
    start_time: float = field(default_factory=time.time)
    errors: int = 0
    block_v_status: str = ""

    @property
    def progress_percent(self) -> float:
        """Overall progress percentage."""
        if self.total_comparisons == 0:
            return 0.0
        return (self.completed_comparisons / self.total_comparisons) * 100

    @property
    def elapsed_minutes(self) -> float:
        """Elapsed time in minutes."""
        return (time.time() - self.start_time) / 60

    @property
    def estimated_remaining_minutes(self) -> float:
        """Estimated remaining time in minutes."""
        if self.completed_comparisons == 0:
            return 0.0
        rate = self.elapsed_minutes / self.completed_comparisons
        remaining = self.total_comparisons - self.completed_comparisons
        return rate * remaining


@dataclass
class EvaluationConfig:
    """Configuration for a single evaluation run."""

    benchmark_version: str = "base"
    max_concurrent: int = 5
    checkpoint_interval: int = 50
    budget_limit_usd: float | None = None
    timeout_seconds: float = 120.0
    auto_resume: bool = True


class Evaluator:
    """Main orchestrator for benchmark evaluation.

    Coordinates all components to run a complete benchmark:
    1. Loads benchmark tasks (Block A, B)
    2. Runs calibration tests (Block A) for reference metrics
    3. Executes pairwise comparisons (Block B) via Swiss tournament
    4. Calculates objective metrics (Block V)
    5. Generates final scores and rankings

    Example:
        >>> evaluator = Evaluator(config)
        >>> evaluator.add_model("gpt-4o", gpt4o_client)
        >>> evaluator.add_model("claude-3.5-sonnet", claude_client)
        >>> evaluator.set_judge(judge_client)
        >>> results = await evaluator.run()
    """

    def __init__(
        self,
        config: Config | None = None,
        eval_config: EvaluationConfig | None = None,
        output_dir: str | Path | None = None,
        elo_registry: Any | None = None,
    ) -> None:
        """Initialize the evaluator.

        Args:
            config: Global configuration.
            eval_config: Evaluation-specific configuration.
            output_dir: Directory for outputs and checkpoints.
            elo_registry: Optional persistent ELO registry for cross-session ratings.
        """
        self._config = config or Config()
        self._eval_config = eval_config or EvaluationConfig(
            benchmark_version=self._config.benchmark_version.value
        )
        self._output_dir = Path(output_dir) if output_dir else Path("results")
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._elo_registry = elo_registry

        self._model_clients: dict[str, BaseModelClient] = {}
        self._judge: BaseJudge | None = None
        self._detectors: dict[str, Any] = {}

        self._pairwise_engine = PairwiseEngine(
            initial_rating=self._config.elo_initial_rating,
            k_factor=self._config.elo_k_factor,
            strategy=PairingStrategy.SWISS,
            registry=elo_registry,
        )

        # Circuit breakers per provider
        self._circuit_breakers: dict[str, CircuitBreaker] = {}

        # Checkpoint manager
        self._checkpoint_manager = CheckpointManager(results_dir=self._output_dir / "checkpoints")

        # Metrics
        self._metrics: MetricsCollector = get_metrics()

        # Progress tracking
        self._progress = EvaluationProgress()

        # Tasks
        self._tasks: list[BenchmarkTask] = []

        # Results storage
        self._block_a_results: dict[str, dict[str, float]] = {}
        self._block_v_results: dict[str, dict[str, float]] = {}

        # Callbacks
        self._on_progress: Callable[[EvaluationProgress], None] | None = None
        self._on_comparison: Callable[[Any], None] | None = None

        # Run state
        self._run_id: str = ""
        self._is_running: bool = False
        self._total_cost_usd: float = 0.0

    def add_model(self, model_id: str, client: BaseModelClient) -> None:
        """Add a model to be evaluated.

        Args:
            model_id: Unique identifier for the model.
            client: Model client instance.
        """
        self._model_clients[model_id] = client
        self._pairwise_engine.register_models([model_id])

        # Create circuit breaker for provider
        provider = self._get_provider(model_id)
        if provider not in self._circuit_breakers:
            self._circuit_breakers[provider] = CircuitBreaker(
                provider=provider,
                failure_threshold=self._config.circuit_breaker_failure_threshold,
                success_threshold=self._config.circuit_breaker_success_threshold,
                timeout_seconds=self._config.circuit_breaker_timeout,
            )

    def set_judge(self, judge: BaseJudge) -> None:
        """Set the judge for pairwise comparisons.

        Args:
            judge: Judge instance (must be calibrated).
        """
        self._judge = judge

    def add_detector(self, name: str, detector: Any) -> None:
        """Add a detector for Block V metrics.

        Args:
            name: Detector name (russism, anglicism, markers, fertility).
            detector: Detector instance.
        """
        self._detectors[name] = detector

    def load_tasks(self, tasks: Sequence[BenchmarkTask]) -> None:
        """Load benchmark tasks.

        Args:
            tasks: List of benchmark tasks.
        """
        self._tasks = list(tasks)
        self._progress.total_tasks = len(tasks)

    def load_tasks_from_file(self, path: str | Path) -> None:
        """Load benchmark tasks from JSON file.

        Args:
            path: Path to benchmark JSON file.
        """
        path = Path(path)
        with path.open() as f:
            data = json.load(f)

        tasks: list[BenchmarkTask] = []

        block_a = data.get("block_a", {})
        for mc in block_a.get("mc", []):
            options = mc.get("options", [])
            options_text = "\n".join(options) if options else ""
            full_prompt = f"{mc['prompt']}\n{options_text}\n\nВідповідь (лише літера):"
            tasks.append(
                BenchmarkTask(
                    id=mc["id"],
                    type="multiple_choice",
                    category=mc.get("category", ""),
                    prompt=full_prompt,
                    reference=mc.get("correct"),
                    metadata={"options": options},
                )
            )

        for gec in block_a.get("gec", []):
            raw_input = gec.get("input", gec.get("prompt", ""))
            gec_prompt = (
                f"Виправте граматичні та стилістичні помилки в реченні. "
                f"Поверніть лише виправлене речення без пояснень.\n\n"
                f"Речення: {raw_input}\n\nВиправлене речення:"
            )
            tasks.append(
                BenchmarkTask(
                    id=gec["id"],
                    type="gec",
                    category=gec.get("category", ""),
                    prompt=gec_prompt,
                    reference=gec.get("expected_output", gec.get("reference")),
                )
            )

        for trans in block_a.get("translation", []):
            source_text = trans.get("source", trans.get("prompt", ""))
            source_lang = trans.get("source_lang", "en")
            lang_name = "англійської" if source_lang == "en" else "російської"
            trans_prompt = (
                f"Перекладіть текст з {lang_name} мови на українську. "
                f"Поверніть лише переклад без пояснень.\n\n"
                f"Текст: {source_text}\n\nПереклад:"
            )
            tasks.append(
                BenchmarkTask(
                    id=trans["id"],
                    type="translation",
                    category=f"{source_lang}-uk",
                    prompt=trans_prompt,
                    reference=trans.get("reference"),
                )
            )

        for fp in block_a.get("false_positive", []):
            raw_text = fp.get("text", fp.get("prompt", ""))
            fp_prompt = (
                f"Перевірте текст на наявність граматичних помилок. "
                f"Якщо помилок немає, напишіть 'CORRECT'. "
                f"Якщо є помилки, виправте їх.\n\n"
                f"Текст: {raw_text}\n\nВідповідь:"
            )
            tasks.append(
                BenchmarkTask(
                    id=fp["id"],
                    type="false_positive",
                    category="false_positive",
                    prompt=fp_prompt,
                    reference=None,
                )
            )

        for pm in block_a.get("positive_marker", []):
            pm_prompt = (
                f"Напишіть речення, використовуючи кличний відмінок або українські "
                f"частки (бо, ж, же, хіба, невже). Контекст: {pm.get('context', '')}\n\n"
                f"Приклад: {pm.get('native_form', '')}\n\nВаше речення:"
            )
            tasks.append(
                BenchmarkTask(
                    id=pm["id"],
                    type="positive_marker",
                    category=pm.get("category", "vocative"),
                    prompt=pm_prompt,
                    reference=pm.get("native_form"),
                    metadata={"marker_regex": pm.get("marker_regex")},
                )
            )

        block_b = data.get("block_b", {})
        for gen in block_b.get("generation", []):
            tasks.append(
                BenchmarkTask(
                    id=gen["id"],
                    type="free_generation",
                    category=gen.get("category", ""),
                    prompt=gen["prompt"],
                )
            )

        for adv in block_b.get("adversarial", []):
            tasks.append(
                BenchmarkTask(
                    id=adv["id"],
                    type="adversarial",
                    category=adv.get("category", ""),
                    prompt=adv["prompt"],
                )
            )

        self.load_tasks(tasks)

    def set_progress_callback(self, callback: Callable[[EvaluationProgress], None]) -> None:
        """Set callback for progress updates.

        Args:
            callback: Function called with progress updates.
        """
        self._on_progress = callback

    def set_comparison_callback(self, callback: Callable[[Any], None]) -> None:
        """Set callback for comparison results.

        Args:
            callback: Function called after each comparison.
        """
        self._on_comparison = callback

    async def run(self) -> list[EvaluationResultData]:
        """Run the complete benchmark evaluation.

        Returns:
            List of evaluation results for all models.

        Raises:
            RuntimeError: If no models or judge configured.
        """
        if not self._model_clients:
            raise RuntimeError("No models configured. Use add_model() first.")
        if self._judge is None:
            raise RuntimeError("No judge configured. Use set_judge() first.")

        self._is_running = True
        self._run_id = self._generate_run_id()
        self._progress = EvaluationProgress(start_time=time.time())

        try:
            # Try to resume from checkpoint if enabled
            if self._eval_config.auto_resume:
                resumed = await self._try_resume()
                if resumed:
                    return await self._continue_evaluation()

            # Fresh evaluation
            return await self._run_evaluation()

        finally:
            self._is_running = False

    async def _run_evaluation(self) -> list[EvaluationResultData]:
        """Run fresh evaluation from start."""
        model_ids = list(self._model_clients.keys())

        # Calculate total comparisons
        num_rounds = self._pairwise_engine.get_recommended_rounds(len(model_ids))
        generation_tasks = [
            t for t in self._tasks if t.type in ("generation", "free_generation", "adversarial")
        ]
        self._progress.total_rounds = num_rounds
        self._progress.total_comparisons = (
            len(model_ids) * (len(model_ids) - 1) // 2 * len(generation_tasks) // num_rounds
        )

        # Run Block A (calibration tests) - reference metrics
        await self._run_block_a()

        # Run Block B (generation tests) - pairwise comparisons
        await self._run_block_b(num_rounds)

        # Run Block V (automatic metrics)
        await self._run_block_v()

        # Calculate final results
        results = self._calculate_results()

        # Save final checkpoint
        await self._save_checkpoint(final=True)

        return results

    async def _continue_evaluation(self) -> list[EvaluationResultData]:
        """Continue evaluation from checkpoint."""
        # Resume from where we left off
        remaining_rounds = self._progress.total_rounds - self._progress.current_round

        if remaining_rounds > 0:
            await self._run_block_b(remaining_rounds)

        # Run Block V if not done
        if not self._block_v_results:
            await self._run_block_v()

        results = self._calculate_results()
        await self._save_checkpoint(final=True)
        return results

    async def _run_block_a(self) -> None:
        """Run Block A calibration tests."""
        block_a_tasks = [
            t
            for t in self._tasks
            if t.type
            in ("multiple_choice", "gec", "translation", "false_positive", "positive_marker")
        ]

        if not block_a_tasks:
            # Initialize with defaults if no Block A tasks
            for model_id in self._model_clients:
                self._block_a_results[model_id] = {
                    "mc_accuracy": 0.0,
                    "gec_f1": 0.0,
                    "translation_comet": 0.0,
                    "false_positive_rate": 0.0,
                }
            return

        for model_id, client in self._model_clients.items():
            results = await self._evaluate_block_a_model(model_id, client, block_a_tasks)
            self._block_a_results[model_id] = results

    async def _evaluate_block_a_model(
        self,
        model_id: str,
        client: BaseModelClient,
        tasks: list[BenchmarkTask],
    ) -> dict[str, float]:
        """Evaluate a single model on Block A tasks.

        Args:
            model_id: Model identifier.
            client: Model client.
            tasks: Block A tasks.

        Returns:
            Dict with mc_accuracy, gec_f1, translation_comet, false_positive_rate.
        """
        mc_correct = 0
        mc_total = 0
        gec_scores: list[float] = []
        gec_total = 0
        translation_scores: list[float] = []
        false_positives = 0
        false_positive_total = 0
        positive_marker_scores: list[float] = []
        task_results: list[dict[str, Any]] = []
        checkpoint_interval = 20

        for i, task in enumerate(tasks):
            try:
                response = await asyncio.wait_for(
                    client.generate(task.prompt, temperature=0.0),
                    timeout=60.0,
                )
                self._total_cost_usd += response.cost_usd

                task_result: dict[str, Any] = {
                    "task_id": task.id,
                    "type": task.type,
                    "response": response.text[:500],
                }

                if task.type == "multiple_choice" and task.reference:
                    mc_total += 1
                    # Extract answer letter more robustly - look for A/B/C/D patterns
                    correct = self._check_mc_answer(response.text, task.reference)
                    if correct:
                        mc_correct += 1
                    task_result["correct"] = correct

                elif task.type == "gec" and task.reference:
                    gec_total += 1
                    # Calculate word-level F1 instead of exact match
                    f1_score = self._calculate_gec_f1(response.text.strip(), task.reference.strip())
                    gec_scores.append(f1_score)
                    task_result["f1_score"] = f1_score
                    task_result["correct"] = f1_score >= 0.9  # For backward compatibility

                elif task.type == "translation" and task.reference:
                    score = self._calculate_translation_similarity(response.text, task.reference)
                    translation_scores.append(score)
                    task_result["score"] = score

                elif task.type == "false_positive":
                    false_positive_total += 1
                    flagged = self._check_false_positive_flagged(response.text)
                    if flagged:
                        false_positives += 1
                    task_result["flagged"] = flagged

                elif task.type == "positive_marker":
                    marker_regex = task.metadata.get("marker_regex")
                    if marker_regex:
                        import re

                        has_marker = bool(re.search(marker_regex, response.text, re.IGNORECASE))
                        positive_marker_scores.append(1.0 if has_marker else 0.0)
                        task_result["has_marker"] = has_marker

                task_results.append(task_result)
                self._progress.completed_tasks += 1
                self._notify_progress()

                if (i + 1) % checkpoint_interval == 0:
                    self._save_block_a_checkpoint(model_id, task_results, i + 1, len(tasks))

            except TimeoutError:
                self._progress.errors += 1
                self._progress.completed_tasks += 1
                self._notify_progress()
                self._metrics.record_error("TimeoutError", self._get_provider(model_id))
                task_results.append({"task_id": task.id, "type": task.type, "error": "timeout"})
            except Exception as e:
                self._progress.errors += 1
                self._progress.completed_tasks += 1
                self._notify_progress()
                self._metrics.record_error(str(type(e).__name__), self._get_provider(model_id))
                task_results.append({"task_id": task.id, "type": task.type, "error": str(e)})

        self._save_block_a_checkpoint(model_id, task_results, len(tasks), len(tasks))

        return {
            "mc_accuracy": mc_correct / mc_total if mc_total > 0 else 0.0,
            "gec_f1": sum(gec_scores) / len(gec_scores) if gec_scores else 0.0,
            "translation_comet": sum(translation_scores) / len(translation_scores)
            if translation_scores
            else 0.0,
            "false_positive_rate": false_positives / false_positive_total
            if false_positive_total > 0
            else 0.0,
            "positive_markers_score": sum(positive_marker_scores) / len(positive_marker_scores)
            if positive_marker_scores
            else 0.0,
        }

    def _save_block_a_checkpoint(
        self, model_id: str, results: list[dict[str, Any]], completed: int, total: int
    ) -> None:
        checkpoint_dir = self._config.data_dir / "checkpoints" / "block_a"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        safe_model_id = model_id.replace("/", "_")
        checkpoint_file = checkpoint_dir / f"{safe_model_id}_checkpoint.json"

        data = {
            "model_id": model_id,
            "completed": completed,
            "total": total,
            "cost_usd": self._total_cost_usd,
            "errors": self._progress.errors,
            "timestamp": datetime.now().isoformat(),
            "results": results,
        }

        temp_file = checkpoint_file.with_suffix(".json.tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        temp_file.rename(checkpoint_file)

    def _check_mc_answer(self, response: str, correct: str) -> bool:
        """Check if MC response contains the correct answer letter."""
        import re

        response_upper = response.upper().strip()
        correct_upper = correct.upper().strip()

        first_char = response_upper[0] if response_upper else ""
        if first_char == correct_upper:
            return True

        pattern = r"\b([A-DА-Г])\b"
        matches = re.findall(pattern, response_upper[:50])
        if matches and matches[0] == correct_upper:
            return True

        return correct_upper in response_upper[:10]

    def _calculate_gec_f1(self, hypothesis: str, reference: str) -> float:
        """Calculate word-level F1 between hypothesis and reference."""
        hyp_words = hypothesis.lower().split()
        ref_words = reference.lower().split()

        if not ref_words:
            return 1.0 if not hyp_words else 0.0
        if not hyp_words:
            return 0.0

        hyp_set = set(hyp_words)
        ref_set = set(ref_words)

        common = len(hyp_set & ref_set)
        precision = common / len(hyp_set) if hyp_set else 0.0
        recall = common / len(ref_set) if ref_set else 0.0

        if precision + recall == 0:
            return 0.0
        f1 = 2 * precision * recall / (precision + recall)

        len_ratio = min(len(hyp_words), len(ref_words)) / max(len(hyp_words), len(ref_words))
        adjusted_f1 = f1 * (0.7 + 0.3 * len_ratio)

        return min(adjusted_f1, 1.0)

    def _calculate_translation_similarity(self, hypothesis: str, reference: str) -> float:
        """Calculate translation similarity score (word overlap with order bonus)."""
        hyp_words = hypothesis.lower().split()
        ref_words = reference.lower().split()

        if not ref_words:
            return 0.0
        if not hyp_words:
            return 0.0

        hyp_set = set(hyp_words)
        ref_set = set(ref_words)
        overlap = len(hyp_set & ref_set) / len(ref_set)

        order_bonus = 0.0
        if len(hyp_words) >= 2 and len(ref_words) >= 2:
            from itertools import pairwise

            bigrams_hyp = set(pairwise(hyp_words))
            bigrams_ref = set(pairwise(ref_words))
            if bigrams_ref:
                order_bonus = len(bigrams_hyp & bigrams_ref) / len(bigrams_ref) * 0.2

        return min(overlap + order_bonus, 1.0)

    def _check_false_positive_flagged(self, response: str) -> bool:
        """Check if model incorrectly flagged valid text as having errors."""
        response_lower = response.lower()

        correct_indicators = [
            "correct",
            "правильн",
            "помилок немає",
            "без помилок",
            "текст правильний",
            "граматично правильн",
            "помилки відсутні",
        ]
        for indicator in correct_indicators:
            if indicator in response_lower:
                return False

        error_indicators = ["помилк", "неправильн", "виправлен", "error", "виправити"]
        return any(indicator in response_lower for indicator in error_indicators)

    async def _run_block_b(self, num_rounds: int) -> None:
        """Run Block B generation tests with pairwise comparisons.

        Args:
            num_rounds: Number of tournament rounds.
        """
        generation_tasks = [
            t
            for t in self._tasks
            if t.type in ("generation", "free_generation", "adversarial", "long_context")
        ]

        if not generation_tasks:
            return

        prompt_ids = [t.id for t in generation_tasks]
        prompt_texts = {t.id: t.prompt for t in generation_tasks}

        logger.info(
            "[BLOCK_B] Starting: current_round=%d, num_rounds=%d, prompts=%d",
            self._progress.current_round,
            num_rounds,
            len(prompt_ids),
        )
        # DEBUG: Print to console
        print(
            f"[BLOCK_B DEBUG] Models in elo_calculator: {list(self._pairwise_engine.elo_calculator.ratings.keys())}"
        )

        for round_num in range(self._progress.current_round + 1, num_rounds + 1):
            self._progress.current_round = round_num

            # Schedule round
            tournament_round = self._pairwise_engine.schedule_round(
                prompt_ids=prompt_ids,
                round_number=round_num,
            )
            logger.info(
                "[BLOCK_B] Round %d: scheduled %d comparisons",
                round_num,
                len(tournament_round.comparisons),
            )
            # DEBUG: Print scheduled pairs
            pairs = set((c.model_a, c.model_b) for c in tournament_round.comparisons)
            print(
                f"[BLOCK_B DEBUG] Round {round_num}: {len(tournament_round.comparisons)} comparisons, pairs: {pairs}"
            )

            # Execute comparisons with concurrency control
            semaphore = asyncio.Semaphore(self._eval_config.max_concurrent)

            async def run_comparison(scheduled: Any, sem: asyncio.Semaphore) -> None:
                async with sem:
                    await self._execute_single_comparison(scheduled, prompt_texts)

            await asyncio.gather(
                *[
                    run_comparison(scheduled, semaphore)
                    for scheduled in tournament_round.comparisons
                ]
            )

            await self._save_checkpoint()
            logger.info(
                "[ROUND %d/%d] completed=%d errors=%d ratings=%s",
                round_num,
                num_rounds,
                self._progress.completed_comparisons,
                self._progress.errors,
                {k: f"{v:.0f}" for k, v in self._pairwise_engine.ratings.items()},
            )

            self._notify_progress()

    async def _execute_single_comparison(
        self,
        scheduled: Any,
        prompt_texts: dict[str, str],
    ) -> None:
        """Execute a single comparison with error handling.

        Args:
            scheduled: Scheduled comparison.
            prompt_texts: Dict mapping prompt_id to text.
        """
        if self._judge is None:
            return

        # Check budget
        if (
            self._eval_config.budget_limit_usd
            and self._total_cost_usd >= self._eval_config.budget_limit_usd
        ):
            return

        try:
            prompt_text = prompt_texts.get(scheduled.prompt_id, "")

            result = await self._pairwise_engine.execute_comparison(
                scheduled=scheduled,
                model_clients=self._model_clients,
                judge=self._judge,
                prompt_text=prompt_text,
            )

            # Update costs
            self._total_cost_usd += result.response_a.cost_usd + result.response_b.cost_usd

            self._pairwise_engine.record_result(result)
            self._progress.completed_comparisons += 1
            self._notify_progress()

            logger.info(
                "[COMPARE] %s vs %s on %s -> winner=%s (completed=%d, errors=%d)",
                scheduled.model_a,
                scheduled.model_b,
                scheduled.prompt_id,
                result.verdict.winner.value,
                self._progress.completed_comparisons,
                self._progress.errors,
            )

            if self._elo_registry is not None:
                self._elo_registry.save()

            self._metrics.record_comparison(
                judge=self._judge.model_id,
                status="success",
                duration_seconds=result.verdict.latency_ms / 1000,
            )

            # Callback
            if self._on_comparison:
                self._on_comparison(result)

        except Exception as e:
            self._progress.errors += 1
            self._metrics.record_error(str(type(e).__name__))
            print(f"[ERROR] {scheduled.model_a} vs {scheduled.model_b}: {type(e).__name__}: {e}")
            logger.error(
                "[COMPARE] %s vs %s on %s FAILED: %s: %s",
                scheduled.model_a,
                scheduled.model_b,
                scheduled.prompt_id,
                type(e).__name__,
                str(e),
            )

    async def _run_block_v(self) -> None:
        """Run Block V automatic metrics using detectors."""
        generation_tasks = [
            t for t in self._tasks if t.type in ("generation", "free_generation", "adversarial")
        ]

        sample_prompts = (
            [t.prompt for t in generation_tasks[:10]]
            if generation_tasks
            else [
                "Поясніть, що таке штучний інтелект простими словами.",
                "Напишіть короткий лист-подяку колезі за допомогу.",
                "Опишіть переваги здорового способу життя.",
            ]
        )

        self._progress.block_v_status = "generating"
        self._notify_progress()

        for model_id, client in self._model_clients.items():
            all_texts: list[str] = []

            for i, prompt in enumerate(sample_prompts):
                self._progress.block_v_status = f"generating {i + 1}/{len(sample_prompts)}"
                self._notify_progress()
                try:
                    response = await asyncio.wait_for(
                        client.generate(prompt),
                        timeout=60.0,
                    )
                    all_texts.append(response.text)
                    self._total_cost_usd += response.cost_usd
                except TimeoutError:
                    self._progress.errors += 1
                except Exception:
                    self._progress.errors += 1

            combined_text = " ".join(all_texts)
            logger.info(
                "[BLOCK_V] %s: generated %d texts, combined_text len=%d",
                model_id,
                len(all_texts),
                len(combined_text),
            )

            # Run detectors
            results: dict[str, float] = {
                "fertility_rate": 1.5,  # Default
                "positive_markers": 0.0,
                "russism_rate": 0.0,
                "anglicism_rate": 0.0,
            }

            if "fertility" in self._detectors:
                try:
                    fertility_result = self._detectors["fertility"].calculate(combined_text)
                    results["fertility_rate"] = fertility_result.fertility_rate
                except Exception as e:
                    logger.warning("[BLOCK_V] %s fertility calculation failed: %s", model_id, e)

            if "russism" in self._detectors:
                try:
                    russism_result = self._detectors["russism"].detect(combined_text)
                    results["russism_rate"] = russism_result.rate_per_1k
                    logger.debug(
                        "[BLOCK_V] %s russism detection: %d matches, rate=%.2f, text_len=%d",
                        model_id,
                        russism_result.count,
                        russism_result.rate_per_1k,
                        len(combined_text),
                    )
                except Exception as e:
                    logger.warning("[BLOCK_V] %s russism detection failed: %s", model_id, e)

            if "anglicism" in self._detectors:
                try:
                    anglicism_result = self._detectors["anglicism"].detect(combined_text)
                    results["anglicism_rate"] = anglicism_result.rate_per_1k
                    logger.debug(
                        "[BLOCK_V] %s anglicism detection: %d matches, rate=%.2f",
                        model_id,
                        anglicism_result.count,
                        anglicism_result.rate_per_1k,
                    )
                except Exception as e:
                    logger.warning("[BLOCK_V] %s anglicism detection failed: %s", model_id, e)

            if "markers" in self._detectors:
                try:
                    markers_result = self._detectors["markers"].detect(combined_text)
                    results["positive_markers"] = markers_result.rate_per_1k
                except Exception as e:
                    logger.warning("[BLOCK_V] %s markers detection failed: %s", model_id, e)

            self._block_v_results[model_id] = results

        self._progress.block_v_status = "done"
        self._notify_progress()

    def _calculate_results(self) -> list[EvaluationResultData]:
        """Calculate final evaluation results for all models.

        Returns:
            List of EvaluationResultData for each model.
        """
        results = []
        rankings = self._pairwise_engine.get_rankings()

        for model_id, elo_rating in rankings:
            # Get Block A scores
            block_a = self._block_a_results.get(model_id, {})
            block_a_scores = BlockAScores(
                mc_accuracy=block_a.get("mc_accuracy", 0.0),
                gec_f1=block_a.get("gec_f1", 0.0),
                translation_comet=block_a.get("translation_comet", 0.0),
                false_positive_rate=block_a.get("false_positive_rate", 0.0),
                positive_markers_score=block_a.get("positive_markers_score", 0.0),
            )

            # Get Block B ELO scores (use overall for all for now)
            block_b_scores = BlockBScores(
                generation_elo=elo_rating,
                adversarial_elo=elo_rating,
                long_context_elo=elo_rating,
            )

            # Get Block V scores
            block_v = self._block_v_results.get(model_id, {})
            block_v_scores = BlockVScores(
                fertility_rate=block_v.get("fertility_rate", 1.5),
                positive_markers=block_v.get("positive_markers", 0.0),
                russism_rate=block_v.get("russism_rate", 0.0),
                anglicism_rate=block_v.get("anglicism_rate", 0.0),
            )

            # Assign badge
            badge = self._assign_badge(elo_rating, block_v_scores)

            # Create model scores
            model_scores = ModelScoreData(
                elo_rating=elo_rating,
                block_a=block_a_scores,
                block_b=block_b_scores,
                block_v=block_v_scores,
                badge=badge,
            )

            # Create metadata
            metadata = EvaluationMetadataData(
                benchmark_version=self._eval_config.benchmark_version,
                dataset_hash=self._calculate_dataset_hash(),
                judge_id=self._judge.model_id if self._judge else "",
                judge_calibration_score=0.90,  # Placeholder
                total_prompts=self._progress.total_tasks,
                total_comparisons=self._progress.completed_comparisons,
                runtime_minutes=self._progress.elapsed_minutes,
                total_cost_usd=self._total_cost_usd,
            )

            # Create result
            result = EvaluationResultData(
                model_id=model_id,
                scores=model_scores,
                metadata=metadata,
                comparisons_count=self._progress.completed_comparisons // len(self._model_clients),
            )
            results.append(result)

        return results

    def _assign_badge(self, elo: float, block_v: BlockVScores) -> Badge:
        """Assign quality badge based on scores.

        Args:
            elo: ELO rating.
            block_v: Block V scores.

        Returns:
            Appropriate badge.
        """
        if elo >= 1650 and block_v.russism_rate < 1.0 and block_v.positive_markers >= 5.0:
            return Badge.GOLD
        elif elo >= 1550 and block_v.russism_rate < 3.0 and block_v.positive_markers >= 3.0:
            return Badge.SILVER
        elif elo >= 1450 and block_v.russism_rate < 5.0:
            return Badge.BRONZE
        elif elo >= 1350 and block_v.russism_rate < 10.0:
            return Badge.CAUTION
        else:
            return Badge.NOT_RECOMMENDED

    async def _save_checkpoint(self, final: bool = False) -> None:
        comparison_history = [
            {
                "comparison_id": r.scheduled.comparison_id,
                "prompt_id": r.scheduled.prompt_id,
                "model_a": r.scheduled.model_a,
                "model_b": r.scheduled.model_b,
                "winner": r.verdict.winner.value,
                "confidence": r.verdict.confidence.value,
                "timestamp": r.timestamp.isoformat(),
            }
            for r in self._pairwise_engine._comparison_history
        ]

        checkpoint_data = {
            "run_id": self._run_id,
            "timestamp": datetime.now().isoformat(),
            "progress": {
                "current_round": self._progress.current_round,
                "total_rounds": self._progress.total_rounds,
                "completed_comparisons": self._progress.completed_comparisons,
                "total_comparisons": self._progress.total_comparisons,
                "errors": self._progress.errors,
            },
            "ratings": self._pairwise_engine.ratings,
            "comparison_history": comparison_history,
            "block_a_results": self._block_a_results,
            "block_v_results": self._block_v_results,
            "total_cost_usd": self._total_cost_usd,
            "final": final,
        }

        block_b_checkpoint_dir = self._config.data_dir / "checkpoints" / "block_b"
        block_b_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        suffix = "final" if final else f"round_{self._progress.current_round}"
        checkpoint_path = block_b_checkpoint_dir / f"{self._run_id}_{suffix}.json"
        temp_path = checkpoint_path.with_suffix(".json.tmp")

        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
        temp_path.rename(checkpoint_path)

        logger.debug("[CHECKPOINT] Saved to %s", checkpoint_path)

    async def _try_resume(self) -> bool:
        """Try to resume from a previous checkpoint.

        Returns:
            True if successfully resumed.
        """
        checkpoints = self._checkpoint_manager.list_checkpoints()
        if not checkpoints:
            return False

        for cp_name in reversed(checkpoints):
            if "final" not in cp_name:
                checkpoint_data = self._checkpoint_manager.load_raw(cp_name)
                if checkpoint_data:
                    self._restore_from_checkpoint(checkpoint_data)
                    return True

        return False

    def _restore_from_checkpoint(self, data: dict[str, Any]) -> None:
        """Restore state from checkpoint data.

        Args:
            data: Checkpoint data dict.
        """
        self._run_id = data.get("run_id", self._run_id)

        progress = data.get("progress", {})
        self._progress.current_round = progress.get("current_round", 0)
        self._progress.total_rounds = progress.get("total_rounds", 0)
        self._progress.completed_comparisons = progress.get("completed_comparisons", 0)
        self._progress.total_comparisons = progress.get("total_comparisons", 0)
        self._progress.errors = progress.get("errors", 0)

        # Restore ratings
        ratings = data.get("ratings", {})
        for model_id, rating in ratings.items():
            if model_id in self._model_clients:
                self._pairwise_engine.elo_calculator.ratings[model_id] = rating

        self._block_a_results = data.get("block_a_results", {})
        self._block_v_results = data.get("block_v_results", {})
        self._total_cost_usd = data.get("total_cost_usd", 0.0)

    def _generate_run_id(self) -> str:
        """Generate unique run ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        models_hash = hashlib.md5(
            ",".join(sorted(self._model_clients.keys())).encode()
        ).hexdigest()[:8]
        return f"{self._eval_config.benchmark_version}_{timestamp}_{models_hash}"

    def _calculate_dataset_hash(self) -> str:
        """Calculate hash of the dataset for reproducibility."""
        task_ids = sorted(t.id for t in self._tasks)
        return hashlib.sha256(",".join(task_ids).encode()).hexdigest()[:16]

    def _get_provider(self, model_id: str) -> str:
        """Extract provider from model ID."""
        if "gpt" in model_id.lower():
            return "openai"
        elif "claude" in model_id.lower():
            return "anthropic"
        elif "gemini" in model_id.lower():
            return "google"
        elif "nebius" in model_id.lower() or "/" in model_id:
            return "nebius"
        else:
            return "unknown"

    def _notify_progress(self) -> None:
        """Notify progress callback if set."""
        if self._on_progress:
            self._on_progress(self._progress)

    def get_current_rankings(self) -> list[dict[str, Any]]:
        """Get current model rankings.

        Returns:
            List of ranking dicts with model_id, rating, wins, losses, ties.
        """
        return self._pairwise_engine.get_leaderboard()

    def get_progress(self) -> EvaluationProgress:
        """Get current evaluation progress.

        Returns:
            Current progress object.
        """
        return self._progress

    def get_statistics(self) -> dict[str, Any]:
        """Get evaluation statistics.

        Returns:
            Dict with various statistics.
        """
        return {
            "run_id": self._run_id,
            "models": list(self._model_clients.keys()),
            "progress": {
                "percent": self._progress.progress_percent,
                "comparisons": f"{self._progress.completed_comparisons}/{self._progress.total_comparisons}",
                "rounds": f"{self._progress.current_round}/{self._progress.total_rounds}",
                "elapsed_minutes": round(self._progress.elapsed_minutes, 1),
                "estimated_remaining": round(self._progress.estimated_remaining_minutes, 1),
                "errors": self._progress.errors,
            },
            "cost_usd": round(self._total_cost_usd, 4),
            "position_bias": self._pairwise_engine.get_position_bias_stats(),
        }

    async def evaluate_model(
        self,
        model_id: str,
    ) -> ModelEvaluationData:
        """Evaluate a single model on Block A + V (no pairwise comparisons).

        This runs calibration tests (Block A) and automatic metrics (Block V).
        Results are saved and can be used later for pairwise comparisons.

        Args:
            model_id: Model to evaluate.

        Returns:
            ModelEvaluationData with Block A and V scores.
        """
        start_time = time.time()

        model_client = self._create_model_client(model_id)
        self.add_model(model_id, model_client)

        benchmark_file = (
            self._config.data_dir / "benchmarks" / f"{self._eval_config.benchmark_version}.json"
        )
        if benchmark_file.exists():
            self.load_tasks_from_file(benchmark_file)
        else:
            self._load_default_tasks()

        self._setup_detectors()
        self._progress.total_tasks = len(
            [
                t
                for t in self._tasks
                if t.type in ("multiple_choice", "gec", "translation", "false_positive")
            ]
        )

        await self._run_block_a()

        try:
            # Block V generates 10 sample texts, each can take up to 60s
            # Total timeout: 10 * 60 + 60 buffer = 660s
            await asyncio.wait_for(self._run_block_v(), timeout=660.0)
        except TimeoutError:
            logger.warning("[BLOCK_V] %s: Block V timed out after 660s", model_id)
        except Exception as e:
            logger.warning("[BLOCK_V] %s: Block V failed: %s: %s", model_id, type(e).__name__, e)

        block_a = self._block_a_results.get(model_id, {})
        block_v = self._block_v_results.get(model_id, {})

        result = ModelEvaluationData(
            model_id=model_id,
            block_a=BlockAScores(
                mc_accuracy=block_a.get("mc_accuracy", 0.0),
                gec_f1=block_a.get("gec_f1", 0.0),
                translation_comet=block_a.get("translation_comet", 0.0),
                false_positive_rate=block_a.get("false_positive_rate", 0.0),
                positive_markers_score=block_a.get("positive_markers_score", 0.0),
            ),
            block_v=BlockVScores(
                fertility_rate=block_v.get("fertility_rate", 1.5),
                positive_markers=block_v.get("positive_markers", 0.0),
                russism_rate=block_v.get("russism_rate", 0.0),
                anglicism_rate=block_v.get("anglicism_rate", 0.0),
            ),
            benchmark_version=self._eval_config.benchmark_version,
            runtime_minutes=(time.time() - start_time) / 60,
            cost_usd=self._total_cost_usd,
        )

        return result

    async def compare_models(
        self,
        model_ids: list[str],
        judge_id: str = "claude-3-5-haiku-latest",
        rounds: int | None = None,
        anchor_count: int = 2,
    ) -> dict[str, float]:
        """Run pairwise comparisons (Block B only) and update ELO ratings.

        If persistent registry is configured, automatically includes anchor
        models for new model calibration.

        Args:
            model_ids: List of model IDs to compare.
            judge_id: Judge model ID.
            rounds: Number of tournament rounds (auto if None).
            anchor_count: Number of anchor models to include for new models.

        Returns:
            Dict mapping model_id to final ELO rating.
        """
        from ukrqualbench.judges import PairwiseJudge

        self._run_id = self._generate_run_id()
        all_model_ids = list(model_ids)

        if self._elo_registry is not None:
            new_models = self._elo_registry.get_new_models(model_ids)
            if new_models and self._elo_registry.model_count > 0:
                anchors = self._elo_registry.get_anchor_models(anchor_count)
                for anchor in anchors:
                    if anchor not in all_model_ids:
                        all_model_ids.append(anchor)

        for model_id in all_model_ids:
            if model_id not in self._model_clients:
                client = self._create_model_client(model_id)
                self.add_model(model_id, client)

        judge_client = self._create_model_client(judge_id)
        judge = PairwiseJudge(judge_client)
        self.set_judge(judge)

        benchmark_file = (
            self._config.data_dir / "benchmarks" / f"{self._eval_config.benchmark_version}.json"
        )
        if benchmark_file.exists():
            self.load_tasks_from_file(benchmark_file)
        else:
            self._load_default_tasks()

        num_models = len(all_model_ids)
        if rounds is None:
            rounds = self._pairwise_engine.get_recommended_rounds(num_models)

        await self._run_block_b(rounds)

        if self._elo_registry is not None:
            self._elo_registry.save()

        return self._pairwise_engine.ratings

    def _create_model_client(self, model_id: str) -> BaseModelClient:
        """Create model client from model ID."""
        from ukrqualbench.models import (
            create_anthropic_client,
            create_google_client,
            create_local_client,
            create_nebius_client,
            create_openai_client,
        )

        model_lower = model_id.lower()

        if "/" in model_id:
            return create_nebius_client(
                model_id=model_id,
                api_key=self._config.nebius_api_key,
                temperature=self._config.temperature,
            )
        elif (
            model_lower.startswith("gpt-")
            or model_lower.startswith("o1")
            or model_lower.startswith("o3")
        ):
            return create_openai_client(
                model_id=model_id,
                api_key=self._config.openai_api_key,
                temperature=self._config.temperature,
            )
        elif model_lower.startswith("claude-"):
            return create_anthropic_client(
                model_id=model_id,
                api_key=self._config.anthropic_api_key,
                temperature=self._config.temperature,
            )
        elif model_lower.startswith("gemini-"):
            return create_google_client(
                model_id=model_id,
                api_key=self._config.google_api_key,
                temperature=self._config.temperature,
            )
        else:
            return create_local_client(
                model_id=model_id,
                base_url=self._config.local_base_url,
                temperature=self._config.temperature,
            )

    def _setup_detectors(self) -> None:
        """Setup default detectors for Block V."""
        from ukrqualbench.detectors import (
            AnglicismDetector,
            FertilityCalculator,
            PositiveMarkerDetector,
            RussismDetector,
        )

        if "russism" not in self._detectors:
            self.add_detector("russism", RussismDetector())
        if "anglicism" not in self._detectors:
            self.add_detector("anglicism", AnglicismDetector())
        if "markers" not in self._detectors:
            self.add_detector("markers", PositiveMarkerDetector())
        if "fertility" not in self._detectors:
            self.add_detector("fertility", FertilityCalculator())

    def _load_default_tasks(self) -> None:
        """Load benchmark tasks using BenchmarkAssembler."""
        from ukrqualbench.datasets import BenchmarkAssembler

        assembler = BenchmarkAssembler(
            data_dir=self._config.data_dir,
            hf_token=self._config.huggingface_token,
        )
        benchmark = assembler.assemble(self._eval_config.benchmark_version)  # type: ignore[arg-type]

        tasks: list[BenchmarkTask] = []

        for mc_task in benchmark.block_a.mc_tasks:
            tasks.append(
                BenchmarkTask(
                    id=mc_task.id,
                    type="multiple_choice",
                    category=mc_task.category,
                    prompt=mc_task.prompt,
                    reference=mc_task.correct,
                    metadata={"options": mc_task.options},
                )
            )

        for gec_task in benchmark.block_a.gec_tasks:
            tasks.append(
                BenchmarkTask(
                    id=gec_task.id,
                    type="gec",
                    category=gec_task.category,
                    prompt=gec_task.input,
                    reference=gec_task.expected_output,
                )
            )

        for trans_task in benchmark.block_a.translation_tasks:
            tasks.append(
                BenchmarkTask(
                    id=trans_task.id,
                    type="translation",
                    category=f"{trans_task.source_lang}-{trans_task.target_lang}",
                    prompt=trans_task.source,
                    reference=trans_task.reference,
                    metadata={"traps": trans_task.traps},
                )
            )

        for fp_task in benchmark.block_a.false_positive_tasks:
            tasks.append(
                BenchmarkTask(
                    id=fp_task.id,
                    type="false_positive",
                    category="false_positive",
                    prompt=fp_task.text,
                    reference=None,
                    metadata={"author": fp_task.author, "is_correct": fp_task.is_correct},
                )
            )

        for gen_task in benchmark.block_b.generation_tasks:
            tasks.append(
                BenchmarkTask(
                    id=gen_task.id,
                    type="free_generation",
                    category=gen_task.category,
                    prompt=gen_task.prompt,
                )
            )

        for adv_task in benchmark.block_b.adversarial_tasks:
            tasks.append(
                BenchmarkTask(
                    id=adv_task.id,
                    type="adversarial",
                    category=adv_task.category,
                    prompt=adv_task.prompt,
                    metadata={"traps": adv_task.traps_in_prompt},
                )
            )

        for lc_task in benchmark.block_b.long_context_tasks:
            tasks.append(
                BenchmarkTask(
                    id=lc_task.id,
                    type="long_context",
                    category=lc_task.category,
                    prompt=str(lc_task.messages),
                    metadata={"total_tokens": lc_task.total_tokens},
                )
            )

        self.load_tasks(tasks)

    def _create_empty_result(self, model_id: str) -> EvaluationResultData:
        """Create empty result for error cases."""
        return EvaluationResultData(
            model_id=model_id,
            scores=ModelScoreData(
                elo_rating=1500.0,
                block_a=BlockAScores(
                    mc_accuracy=0.0,
                    gec_f1=0.0,
                    translation_comet=0.0,
                    false_positive_rate=0.0,
                    positive_markers_score=0.0,
                ),
                block_b=BlockBScores(
                    generation_elo=1500.0,
                    adversarial_elo=1500.0,
                    long_context_elo=1500.0,
                ),
                block_v=BlockVScores(
                    fertility_rate=1.5,
                    positive_markers=0.0,
                    russism_rate=0.0,
                    anglicism_rate=0.0,
                ),
                badge=Badge.NONE,
            ),
            metadata=EvaluationMetadataData(
                benchmark_version=self._eval_config.benchmark_version,
                dataset_hash="",
                judge_id="",
                judge_calibration_score=0.0,
                total_prompts=0,
                total_comparisons=0,
                runtime_minutes=0.0,
                total_cost_usd=0.0,
            ),
            comparisons_count=0,
        )


def create_evaluator(
    benchmark_version: str = "base",
    output_dir: str | Path | None = None,
    max_concurrent: int = 5,
    budget_limit_usd: float | None = None,
) -> Evaluator:
    """Factory function to create an evaluator.

    Args:
        benchmark_version: Benchmark version (lite/base/large).
        output_dir: Output directory.
        max_concurrent: Max concurrent API calls.
        budget_limit_usd: Optional budget limit.

    Returns:
        Configured Evaluator instance.
    """
    from ukrqualbench.core.config import BenchmarkVersion

    version = BenchmarkVersion(benchmark_version)
    config = Config(benchmark_version=version)
    eval_config = EvaluationConfig(
        benchmark_version=benchmark_version,
        max_concurrent=max_concurrent,
        budget_limit_usd=budget_limit_usd,
    )
    return Evaluator(config=config, eval_config=eval_config, output_dir=output_dir)
