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
    ModelScoreData,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ukrqualbench.detectors.base import BaseDetector
    from ukrqualbench.judges.base import BaseJudge
    from ukrqualbench.models.base import BaseModelClient


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
    ) -> None:
        """Initialize the evaluator.

        Args:
            config: Global configuration.
            eval_config: Evaluation-specific configuration.
            output_dir: Directory for outputs and checkpoints.
        """
        self._config = config or Config()
        self._eval_config = eval_config or EvaluationConfig()
        self._output_dir = Path(output_dir) if output_dir else Path("results")
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Model clients
        self._model_clients: dict[str, BaseModelClient] = {}

        # Judge
        self._judge: BaseJudge | None = None

        # Detectors for Block V
        self._detectors: dict[str, BaseDetector] = {}

        # Pairwise engine
        self._pairwise_engine = PairwiseEngine(
            initial_rating=self._config.elo_initial_rating,
            k_factor=self._config.elo_k_factor,
            strategy=PairingStrategy.SWISS,
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

        tasks = []
        for task_data in data.get("tasks", []):
            task = BenchmarkTask(
                id=task_data["id"],
                type=task_data["type"],
                category=task_data.get("category", ""),
                prompt=task_data["prompt"],
                reference=task_data.get("reference"),
                metadata=task_data.get("metadata", {}),
            )
            tasks.append(task)

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

        # Run each model on Block A tasks
        for model_id, client in self._model_clients.items():
            results = await self._evaluate_block_a_model(model_id, client, block_a_tasks)
            self._block_a_results[model_id] = results
            self._progress.completed_tasks += len(block_a_tasks)
            self._notify_progress()

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
        gec_correct = 0
        gec_total = 0
        translation_scores: list[float] = []
        false_positives = 0
        false_positive_total = 0

        for task in tasks:
            try:
                response = await client.generate(
                    task.prompt,
                    temperature=0.0,
                    max_tokens=512,
                )
                self._total_cost_usd += response.cost_usd

                if task.type == "multiple_choice" and task.reference:
                    mc_total += 1
                    # Check if response contains correct answer
                    if task.reference.upper() in response.text.upper()[:10]:
                        mc_correct += 1

                elif task.type == "gec" and task.reference:
                    gec_total += 1
                    # Simple exact match for now
                    if response.text.strip() == task.reference.strip():
                        gec_correct += 1

                elif task.type == "translation" and task.reference:
                    # Simplified COMET approximation
                    score = self._simple_translation_score(response.text, task.reference)
                    translation_scores.append(score)

                elif task.type == "false_positive":
                    false_positive_total += 1
                    # Check if model incorrectly flagged correct text
                    if "помилк" in response.text.lower() or "error" in response.text.lower():
                        false_positives += 1

            except Exception as e:
                self._progress.errors += 1
                self._metrics.record_error(str(type(e).__name__), self._get_provider(model_id))

        return {
            "mc_accuracy": mc_correct / mc_total if mc_total > 0 else 0.0,
            "gec_f1": gec_correct / gec_total if gec_total > 0 else 0.0,
            "translation_comet": sum(translation_scores) / len(translation_scores)
            if translation_scores
            else 0.0,
            "false_positive_rate": false_positives / false_positive_total
            if false_positive_total > 0
            else 0.0,
        }

    def _simple_translation_score(self, hypothesis: str, reference: str) -> float:
        """Simple translation quality score (placeholder for COMET).

        Args:
            hypothesis: Model translation.
            reference: Reference translation.

        Returns:
            Score between 0 and 1.
        """
        # Very simple overlap-based score
        hyp_words = set(hypothesis.lower().split())
        ref_words = set(reference.lower().split())
        if not ref_words:
            return 0.0
        overlap = len(hyp_words & ref_words) / len(ref_words)
        return min(overlap * 1.2, 1.0)  # Scale up slightly

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

        for round_num in range(self._progress.current_round + 1, num_rounds + 1):
            self._progress.current_round = round_num

            # Schedule round
            tournament_round = self._pairwise_engine.schedule_round(
                prompt_ids=prompt_ids,
                round_number=round_num,
            )

            # Execute comparisons with concurrency control
            semaphore = asyncio.Semaphore(self._eval_config.max_concurrent)

            async def run_comparison(scheduled: Any, sem: asyncio.Semaphore) -> None:
                async with sem:
                    await self._execute_single_comparison(scheduled, prompt_texts)

            # Run comparisons concurrently
            await asyncio.gather(
                *[
                    run_comparison(scheduled, semaphore)
                    for scheduled in tournament_round.comparisons
                ]
            )

            # Checkpoint after each round
            if self._progress.completed_comparisons % self._eval_config.checkpoint_interval == 0:
                await self._save_checkpoint()

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

            # Record result
            self._pairwise_engine.record_result(result)
            self._progress.completed_comparisons += 1

            # Record metrics
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

    async def _run_block_v(self) -> None:
        """Run Block V automatic metrics using detectors."""
        # For each model, generate sample texts and run detectors
        sample_prompts = [
            "Поясніть, що таке штучний інтелект простими словами.",
            "Напишіть короткий лист-подяку колезі за допомогу.",
            "Опишіть переваги здорового способу життя.",
        ]

        for model_id, client in self._model_clients.items():
            all_texts: list[str] = []

            # Generate sample texts
            for prompt in sample_prompts:
                try:
                    response = await client.generate(prompt, max_tokens=300)
                    all_texts.append(response.text)
                    self._total_cost_usd += response.cost_usd
                except Exception:
                    pass

            combined_text = " ".join(all_texts)

            # Run detectors
            results: dict[str, float] = {
                "fertility_rate": 1.5,  # Default
                "positive_markers": 0.0,
                "russism_rate": 0.0,
                "anglicism_rate": 0.0,
            }

            if "fertility" in self._detectors:
                try:
                    fertility_result = self._detectors["fertility"].detect(combined_text)
                    results["fertility_rate"] = fertility_result.metadata.get("fertility_rate", 1.5)
                except Exception:
                    pass

            if "russism" in self._detectors:
                try:
                    russism_result = self._detectors["russism"].detect(combined_text)
                    results["russism_rate"] = russism_result.rate_per_1k
                except Exception:
                    pass

            if "anglicism" in self._detectors:
                try:
                    anglicism_result = self._detectors["anglicism"].detect(combined_text)
                    results["anglicism_rate"] = anglicism_result.rate_per_1k
                except Exception:
                    pass

            if "markers" in self._detectors:
                try:
                    markers_result = self._detectors["markers"].detect(combined_text)
                    results["positive_markers"] = markers_result.rate_per_1k
                except Exception:
                    pass

            self._block_v_results[model_id] = results

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
        """Save checkpoint to disk.

        Args:
            final: Whether this is the final checkpoint.
        """
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
            "block_a_results": self._block_a_results,
            "block_v_results": self._block_v_results,
            "total_cost_usd": self._total_cost_usd,
            "final": final,
        }

        suffix = "final" if final else f"round_{self._progress.current_round}"
        self._checkpoint_manager.save_raw(f"{self._run_id}_{suffix}", checkpoint_data)

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
        judge_id: str = "claude-3-5-haiku-latest",
        resume: bool = True,
    ) -> EvaluationResultData:
        """Evaluate a single model (convenience method).

        Args:
            model_id: Model to evaluate.
            judge_id: Judge model ID.
            resume: Whether to resume from checkpoint.

        Returns:
            Evaluation result for the model.
        """
        from ukrqualbench.judges import PairwiseJudge

        # Create model client
        model_client = self._create_model_client(model_id)
        self.add_model(model_id, model_client)

        # Create judge
        judge_client = self._create_model_client(judge_id)
        judge = PairwiseJudge(judge_client)
        self.set_judge(judge)

        # Load benchmark tasks
        benchmark_file = (
            self._config.data_dir / "benchmarks" / f"{self._eval_config.benchmark_version}.json"
        )
        if benchmark_file.exists():
            self.load_tasks_from_file(benchmark_file)
        else:
            self._load_default_tasks()

        # Setup detectors
        self._setup_detectors()

        # Configure resume
        self._eval_config.auto_resume = resume

        # Run evaluation
        results = await self.run()
        return results[0] if results else self._create_empty_result(model_id)

    async def compare_models(
        self,
        model_ids: list[str],
        judge_id: str = "claude-3-5-haiku-latest",
        rounds: int | None = None,
    ) -> list[EvaluationResultData]:
        """Compare multiple models (convenience method).

        Args:
            model_ids: List of model IDs to compare.
            judge_id: Judge model ID.
            rounds: Number of tournament rounds (auto if None).

        Returns:
            List of evaluation results for all models.
        """

        from ukrqualbench.judges import PairwiseJudge

        # Create model clients
        for model_id in model_ids:
            client = self._create_model_client(model_id)
            self.add_model(model_id, client)

        # Create judge
        judge_client = self._create_model_client(judge_id)
        judge = PairwiseJudge(judge_client)
        self.set_judge(judge)

        # Load benchmark tasks
        benchmark_file = (
            self._config.data_dir / "benchmarks" / f"{self._eval_config.benchmark_version}.json"
        )
        if benchmark_file.exists():
            self.load_tasks_from_file(benchmark_file)
        else:
            self._load_default_tasks()

        # Setup detectors
        self._setup_detectors()

        # Override rounds if specified
        if rounds is not None:
            self._progress.total_rounds = rounds

        # Run evaluation
        return await self.run()

    def _create_model_client(self, model_id: str) -> BaseModelClient:
        """Create model client from model ID."""
        from ukrqualbench.models import (
            create_anthropic_client,
            create_google_client,
            create_nebius_client,
            create_ollama_client,
            create_openai_client,
        )

        model_lower = model_id.lower()

        if (
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
        elif "/" in model_id:
            return create_nebius_client(
                model_id=model_id,
                api_key=self._config.nebius_api_key,
                temperature=self._config.temperature,
            )
        else:
            return create_ollama_client(
                model_id=model_id,
                base_url=self._config.ollama_base_url,
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
        """Load default synthetic tasks for testing."""
        tasks = [
            BenchmarkTask(
                id="gen_1",
                type="generation",
                category="explanation",
                prompt="Поясніть, що таке машинне навчання.",
            ),
            BenchmarkTask(
                id="gen_2",
                type="generation",
                category="advice",
                prompt="Дайте поради щодо здорового харчування.",
            ),
            BenchmarkTask(
                id="gen_3",
                type="generation",
                category="creative",
                prompt="Напишіть короткий вірш про весну.",
            ),
        ]
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
