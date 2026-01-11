"""CLI entry point for UkrQualBench."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ukrqualbench import __version__
from ukrqualbench.core.config import BenchmarkVersion, Config

if TYPE_CHECKING:
    pass

app = typer.Typer(
    name="ukrqualbench",
    help="Benchmark for evaluating Ukrainian language quality in LLMs",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


def create_model_client(model_id: str, config: Config | None = None) -> Any:
    """Create model client from model ID string."""
    from ukrqualbench.models import (
        create_anthropic_client,
        create_google_client,
        create_local_client,
        create_nebius_client,
        create_openai_client,
    )

    config = config or Config()
    model_lower = model_id.lower()

    if "/" in model_id:
        return create_nebius_client(
            model_id=model_id,
            api_key=config.nebius_api_key,
            temperature=config.temperature,
            max_retries=config.max_retries,
        )
    elif model_lower.startswith(("gpt-", "o1", "o3")):
        return create_openai_client(
            model_id=model_id,
            api_key=config.openai_api_key,
            temperature=config.temperature,
            max_retries=config.max_retries,
        )
    elif model_lower.startswith("claude-"):
        return create_anthropic_client(
            model_id=model_id,
            api_key=config.anthropic_api_key,
            temperature=config.temperature,
            max_retries=config.max_retries,
        )
    elif model_lower.startswith("gemini-"):
        return create_google_client(
            model_id=model_id,
            api_key=config.google_api_key,
            temperature=config.temperature,
            max_retries=config.max_retries,
        )
    else:
        return create_local_client(
            model_id=model_id,
            base_url=config.local_base_url,
            temperature=config.temperature,
        )


def version_callback(value: bool) -> None:
    if value:
        rprint(f"[bold blue]UkrQualBench[/bold blue] v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            "-V",
            help="Show version and exit",
            callback=version_callback,
            is_eager=True,
        ),
    ] = None,
) -> None:
    """UkrQualBench: Benchmark for evaluating Ukrainian language quality in LLMs."""


@app.command()
def calibrate(
    judge: Annotated[str, typer.Option("--judge", "-j", help="Judge model to calibrate")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Output directory")] = Path(
        "results/calibration"
    ),
    data_dir: Annotated[
        Path, typer.Option("--data-dir", "-d", help="Calibration data directory")
    ] = Path("data/gold"),
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show detailed output")] = False,
) -> None:
    """Calibrate a judge model before evaluation."""
    from ukrqualbench.judges import CalibrationTask, JudgeCalibrator
    from ukrqualbench.reports import generate_calibration_html

    config = Config()
    output.mkdir(parents=True, exist_ok=True)

    rprint(
        Panel.fit(
            f"[bold]Calibrating Judge[/bold]\n\nModel: [cyan]{judge}[/cyan]\nOutput: [dim]{output}[/dim]",
            title="UkrQualBench Calibration",
        )
    )

    calibration_tasks: list[CalibrationTask] = []

    task_files = {
        "multiple_choice": data_dir / "mc_calibration.json",
        "gec": data_dir / "gec_calibration.json",
        "russism": data_dir / "russism_calibration.json",
        "false_positive": data_dir / "false_positive_calibration.json",
        "pairwise": data_dir / "pairwise_calibration.json",
    }

    with console.status("[cyan]Loading calibration data..."):
        for task_type, filepath in task_files.items():
            if filepath.exists():
                with open(filepath) as f:
                    data = json.load(f)
                    for item in data.get("tasks", []):
                        calibration_tasks.append(
                            CalibrationTask(
                                id=item.get("id", f"{task_type}_{len(calibration_tasks)}"),
                                task_type=task_type,
                                input_data=item.get("input", {}),
                                expected_output=item.get("expected"),
                                metadata=item.get("metadata", {}),
                            )
                        )

    if not calibration_tasks:
        rprint("[yellow]Warning: No calibration data found. Using synthetic tests.[/yellow]")
        calibration_tasks = _create_synthetic_calibration_data()

    rprint(f"[green]Loaded {len(calibration_tasks)} calibration tasks[/green]")

    judge_client = create_model_client(judge, config)

    calibrator = JudgeCalibrator(
        model=judge_client,
        thresholds={
            "mc_accuracy": 0.85,
            "gec_f1": 0.80,
            "russism_f1": 0.85,
            "false_positive": 0.15,
            "pairwise_consistency": 0.90,
            "position_bias": 0.05,
            "length_bias": 0.30,
            "final_score": 0.80,
        },
    )

    def on_progress(current: int, total: int, task_type: str) -> None:
        rprint(f"  [{current}/{total}] {task_type}", end="\r")

    async def run_calibration() -> Any:
        return await calibrator.calibrate(calibration_tasks, on_progress=on_progress)

    rprint(f"[cyan]Running calibration on {judge}...[/cyan]")
    result = asyncio.run(run_calibration())
    rprint()

    _display_calibration_results(result, verbose)

    result_file = output / f"{judge.replace('/', '_')}_calibration.json"
    with open(result_file, "w") as f:
        json.dump(
            {
                "judge_id": result.judge_id,
                "passed": result.passed,
                "mc_accuracy": result.mc_accuracy,
                "gec_f1": result.gec_f1,
                "russism_f1": result.russism_f1,
                "false_positive_rate": result.false_positive_rate,
                "pairwise_consistency": result.pairwise_consistency,
                "position_bias": result.position_bias,
                "length_bias_correlation": result.length_bias_correlation,
                "final_score": result.final_score,
                "failure_reasons": result.failure_reasons,
                "timestamp": result.timestamp.isoformat(),
            },
            f,
            indent=2,
        )

    rprint(f"\n[dim]Results saved to: {result_file}[/dim]")

    html_file = output / f"{judge.replace('/', '_')}_calibration.html"
    html_content = generate_calibration_html(result)
    with open(html_file, "w") as f:
        f.write(html_content)
    rprint(f"[dim]HTML report saved to: {html_file}[/dim]")

    if not result.passed:
        raise typer.Exit(1)


def _display_calibration_results(result: Any, verbose: bool) -> None:
    table = Table(title="Calibration Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Threshold", style="yellow")
    table.add_column("Status", style="bold")

    metrics = [
        ("MC Accuracy", result.mc_accuracy, 0.85, result.mc_accuracy >= 0.85),
        ("GEC F1", result.gec_f1, 0.80, result.gec_f1 >= 0.80),
        ("Russism F1", result.russism_f1, 0.85, result.russism_f1 >= 0.85),
        (
            "False Positive Rate",
            result.false_positive_rate,
            0.15,
            result.false_positive_rate <= 0.15,
        ),
        (
            "Pairwise Consistency",
            result.pairwise_consistency,
            0.90,
            result.pairwise_consistency >= 0.90,
        ),
        ("Position Bias", result.position_bias, 0.05, result.position_bias <= 0.05),
        (
            "Length Bias",
            result.length_bias_correlation,
            0.30,
            result.length_bias_correlation <= 0.30,
        ),
    ]

    for name, value, threshold, passed in metrics:
        status = "[green]\u2713[/green]" if passed else "[red]\u2717[/red]"
        table.add_row(name, f"{value:.2%}", f"{threshold:.0%}", status)

    console.print(table)
    rprint(f"\n[bold]Final Score:[/bold] {result.final_score:.2%}")

    if result.passed:
        rprint(
            Panel("[bold green]PASSED[/bold green] - Judge is ready for evaluation", title="Result")
        )
    else:
        rprint(
            Panel(
                "[bold red]FAILED[/bold red]\n\nReasons:\n"
                + "\n".join(f"  \u2022 {r}" for r in result.failure_reasons),
                title="Result",
            )
        )


def _create_synthetic_calibration_data() -> list[Any]:
    from ukrqualbench.judges import CalibrationTask

    tasks: list[CalibrationTask] = []

    for i in range(10):
        tasks.append(
            CalibrationTask(
                id=f"mc_{i}",
                task_type="multiple_choice",
                input_data={
                    "question": f"Test question {i}",
                    "options": ["A) Option A", "B) Option B", "C) Option C", "D) Option D"],
                },
                expected_output="A",
            )
        )

    for i in range(10):
        tasks.append(
            CalibrationTask(
                id=f"gec_{i}",
                task_type="gec",
                input_data={
                    "original": "Це є тестове речення.",
                    "reference": "Це тестове речення.",
                },
                expected_output=["є → видалити"],
            )
        )

    for i in range(10):
        tasks.append(
            CalibrationTask(
                id=f"russism_{i}",
                task_type="russism",
                input_data={"text": "Він прийняв участь у міроприємстві."},
                expected_output=["прийняв участь", "міроприємстві"],
            )
        )

    for i in range(5):
        tasks.append(
            CalibrationTask(
                id=f"fp_{i}",
                task_type="false_positive",
                input_data={"text": "Пане Андрію, це чудова ідея!"},
                expected_output=True,
            )
        )

    for i in range(10):
        tasks.append(
            CalibrationTask(
                id=f"pairwise_{i}",
                task_type="pairwise",
                input_data={
                    "prompt": "Поясни, що таке машинне навчання.",
                    "response_a": "Машинне навчання — це галузь штучного інтелекту.",
                    "response_b": "Machine learning - це когда компьютер учится сам.",
                },
                expected_output="A",
            )
        )

    return tasks


@app.command()
def evaluate(
    model: Annotated[str, typer.Option("--model", "-m", help="Model to evaluate")],
    benchmark: Annotated[
        BenchmarkVersion, typer.Option("--benchmark", "-b", help="Benchmark version")
    ] = BenchmarkVersion.LITE,
    output: Annotated[Path, typer.Option("--output", "-o", help="Output directory")] = Path(
        "results/evaluations"
    ),
    resume: Annotated[bool, typer.Option("--resume", help="Resume from checkpoint")] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show detailed LLM request/response logs")
    ] = False,
) -> None:
    """Evaluate a single model on Block A + V (calibration tests and automatic metrics).

    This does NOT run pairwise comparisons. Use 'compare' command for ELO ratings.
    Results are saved and can be loaded by 'compare' or 'leaderboard' commands.
    """
    import logging

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"evaluate_{model.replace('/', '_')}_{benchmark.value}.log"

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    model_logger = logging.getLogger("ukrqualbench.models")
    model_logger.setLevel(logging.INFO)
    model_logger.addHandler(file_handler)

    if verbose:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
        )
        model_logger.addHandler(console_handler)

    rprint(f"[dim]Logging to: {log_file}[/dim]")

    from ukrqualbench.core.evaluator import EvaluationProgress, Evaluator
    from ukrqualbench.core.schemas import ModelEvaluationData

    config = Config()
    config.benchmark_version = benchmark

    output.mkdir(parents=True, exist_ok=True)

    checkpoint_file = (
        config.data_dir / "checkpoints" / "block_a" / f"{model.replace('/', '_')}_checkpoint.json"
    )
    if resume and checkpoint_file.exists():
        with open(checkpoint_file) as f:
            checkpoint = json.load(f)
        rprint(
            f"[yellow]Resuming from checkpoint: {checkpoint['completed']}/{checkpoint['total']} tasks[/yellow]"
        )
    elif resume:
        rprint("[yellow]No checkpoint found, starting fresh[/yellow]")

    rprint(
        Panel.fit(
            f"[bold]Evaluating Model (Block A + V)[/bold]\n\n"
            f"Model: [cyan]{model}[/cyan]\n"
            f"Benchmark: [yellow]{benchmark.value}[/yellow]\n\n"
            f"[dim]This runs calibration tests and automatic metrics.\n"
            f"For ELO ratings, use 'compare' command after evaluation.[/dim]",
            title="UkrQualBench Evaluation",
        )
    )

    evaluator = Evaluator(config=config)

    from rich.live import Live
    from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
    from rich.table import Table as ProgressTable

    block_a_progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[cyan]{task.fields[status]}[/cyan]"),
    )
    block_v_status = "[dim]Waiting...[/dim]"

    def make_progress_table() -> ProgressTable:
        table = ProgressTable.grid(padding=(0, 1))
        table.add_row(block_a_progress)
        table.add_row(f"  Block V (metrics): {block_v_status}")
        return table

    task_id = None

    def on_progress(p: EvaluationProgress) -> None:
        nonlocal task_id, block_v_status
        if p.total_tasks > 0:
            if task_id is None:
                task_id = block_a_progress.add_task(
                    "[bold]Block A[/bold]",
                    total=p.total_tasks,
                    status=f"${evaluator._total_cost_usd:.4f}",
                )
            block_a_progress.update(
                task_id,
                completed=p.completed_tasks,
                status=f"${evaluator._total_cost_usd:.4f} | {p.errors} errors",
            )
        if p.block_v_status:
            block_v_status = f"[yellow]{p.block_v_status}[/yellow]"
            if live_display:
                live_display.update(make_progress_table())

    evaluator.set_progress_callback(on_progress)

    rprint(f"[cyan]Running Block A + V evaluation for {model}...[/cyan]")

    live_display: Live | None = None

    async def run_with_progress() -> ModelEvaluationData:
        nonlocal block_v_status, live_display
        result = await evaluator.evaluate_model(model_id=model)
        block_v_status = "[green]Done[/green]"
        if live_display:
            live_display.update(make_progress_table())
        return result

    with Live(make_progress_table(), console=console, refresh_per_second=10) as live:
        live_display = live
        result = asyncio.run(run_with_progress())
    rprint()

    _display_model_evaluation(result)

    result_file = output / f"{model.replace('/', '_')}.json"
    with open(result_file, "w") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)

    rprint(f"\n[green]Results saved to: {result_file}[/green]")
    rprint(f"[dim]Run 'ukrqualbench compare --models {model},...' for ELO ratings[/dim]")


def _display_model_evaluation(result: Any) -> None:
    from ukrqualbench.core.schemas import ModelEvaluationData

    if not isinstance(result, ModelEvaluationData):
        return

    table = Table(title=f"Evaluation Results: {result.model_id}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("[bold]Block A (Calibration)[/bold]", "")
    table.add_row("  MC Accuracy", f"{result.block_a.mc_accuracy:.1%}")
    table.add_row("  GEC F1", f"{result.block_a.gec_f1:.1%}")
    table.add_row("  Translation COMET", f"{result.block_a.translation_comet:.3f}")
    table.add_row("  False Positive Rate", f"{result.block_a.false_positive_rate:.1%}")

    table.add_row("[bold]Block V (Automatic Metrics)[/bold]", "")
    table.add_row("  Fertility Rate", f"{result.block_v.fertility_rate:.2f}")
    table.add_row("  Positive Markers", f"{result.block_v.positive_markers:.1f}/1K")
    table.add_row("  Russism Rate", f"{result.block_v.russism_rate:.2f}/1K")
    table.add_row("  Anglicism Rate", f"{result.block_v.anglicism_rate:.2f}/1K")

    table.add_row("", "")
    table.add_row("[dim]Runtime[/dim]", f"[dim]{result.runtime_minutes:.1f} min[/dim]")
    table.add_row("[dim]Cost[/dim]", f"[dim]${result.cost_usd:.4f}[/dim]")

    console.print(table)


def _display_elo_results(elo_ratings: dict[str, float], registry: Any) -> None:
    from ukrqualbench.core.elo_registry import ELORegistry

    table = Table(title="ELO Ratings (Block B Comparison)")
    table.add_column("Rank", style="dim", justify="right")
    table.add_column("Model", style="cyan")
    table.add_column("ELO", style="green", justify="right")
    table.add_column("Games", justify="right")
    table.add_column("W/L/T", justify="center")
    table.add_column("Win Rate", justify="right")
    table.add_column("Status", justify="center")

    sorted_ratings = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)

    for rank, (model_id, rating) in enumerate(sorted_ratings, start=1):
        games = "-"
        wlt = "-"
        win_rate = "-"
        status = ""

        if isinstance(registry, ELORegistry):
            entry = registry.get_model(model_id)
            if entry:
                games = str(entry.games_played)
                wlt = f"{entry.wins}/{entry.losses}/{entry.ties}"
                win_rate = f"{entry.win_rate:.1%}"
                status = (
                    "[dim]provisional[/dim]" if entry.is_provisional else "[green]stable[/green]"
                )

        table.add_row(
            str(rank),
            model_id,
            f"{rating:.1f}",
            games,
            wlt,
            win_rate,
            status,
        )

    console.print(table)


@app.command()
def compare(
    models: Annotated[str, typer.Option("--models", "-m", help="Comma-separated models")],
    benchmark: Annotated[
        BenchmarkVersion, typer.Option("--benchmark", "-b", help="Benchmark version")
    ] = BenchmarkVersion.LITE,
    judge: Annotated[
        str, typer.Option("--judge", "-j", help="Judge model")
    ] = "claude-3-5-haiku-latest",
    rounds: Annotated[int | None, typer.Option("--rounds", help="Tournament rounds")] = None,
    output: Annotated[Path, typer.Option("--output", "-o", help="Output directory")] = Path(
        "results"
    ),
    max_cost: Annotated[float | None, typer.Option("--max-cost", help="Max cost USD")] = None,
    persistent: Annotated[
        bool, typer.Option("--persistent/--no-persistent", help="Use persistent ELO registry")
    ] = True,
) -> None:
    """Compare multiple models using pairwise evaluation."""
    import math

    from ukrqualbench.core.elo_registry import ELORegistry
    from ukrqualbench.core.evaluator import EvaluationProgress, Evaluator

    config = Config()
    config.benchmark_version = benchmark
    if max_cost is not None:
        config.max_cost_usd = max_cost

    model_list = [m.strip() for m in models.split(",")]
    n_models = len(model_list)

    if rounds is None:
        rounds = math.ceil(math.log2(max(n_models, 2))) + 2

    output.mkdir(parents=True, exist_ok=True)

    registry: ELORegistry | None = None
    if persistent:
        registry_path = config.data_dir / "elo_registry.json"
        registry = ELORegistry(
            registry_path=registry_path,
            initial_rating=config.elo_initial_rating,
            k_factor=config.elo_k_factor,
        )
        new_models = registry.get_new_models(model_list)
        existing_models = registry.get_existing_models(model_list)

        status_lines = [f"[bold]Comparing Models[/bold]\n\nModels: [cyan]{n_models}[/cyan]"]
        for m in model_list:
            if m in new_models:
                status_lines.append(f"  \u2022 {m} [yellow](new)[/yellow]")
            else:
                entry = registry.get_model(m)
                rating = entry.rating if entry else config.elo_initial_rating
                status_lines.append(f"  \u2022 {m} [dim](ELO: {rating:.0f})[/dim]")

        if existing_models and new_models:
            anchors = registry.get_anchor_models(min(3, len(existing_models)))
            if anchors:
                status_lines.append(
                    f"\n[dim]Anchor models for calibration: {', '.join(anchors)}[/dim]"
                )

        status_lines.append(f"\nBenchmark: [yellow]{benchmark.value}[/yellow]")
        status_lines.append(f"Rounds: [yellow]{rounds}[/yellow]")
        status_lines.append(f"Registry: [dim]{registry_path}[/dim]")
    else:
        status_lines = [
            f"[bold]Comparing Models[/bold]\n\nModels: [cyan]{n_models}[/cyan]",
            *[f"  \u2022 {m}" for m in model_list],
            f"\nBenchmark: [yellow]{benchmark.value}[/yellow]",
            f"Rounds: [yellow]{rounds}[/yellow]",
        ]

    rprint(Panel.fit("\n".join(status_lines), title="UkrQualBench Comparison"))

    evaluator = Evaluator(config=config, elo_registry=registry)

    def on_progress(p: EvaluationProgress) -> None:
        if p.total_comparisons > 0:
            rprint(
                f"  [Round {p.current_round}/{p.total_rounds}] "
                f"{p.completed_comparisons}/{p.total_comparisons} comparisons "
                f"({p.progress_percent:.0f}%)",
                end="\r",
            )

    evaluator.set_progress_callback(on_progress)

    async def run_comparison() -> dict[str, float]:
        return await evaluator.compare_models(model_ids=model_list, judge_id=judge, rounds=rounds)

    rprint(f"[cyan]Running {rounds} rounds of comparison (Block B only)...[/cyan]")
    elo_ratings = asyncio.run(run_comparison())
    rprint()

    if registry:
        rprint(
            f"[dim]ELO registry saved: {registry.model_count} models, "
            f"{registry.comparison_count} comparisons[/dim]"
        )

    _display_elo_results(elo_ratings, registry)

    results_file = output / "comparison_results.json"
    with open(results_file, "w") as f:
        json.dump(
            {
                "benchmark_version": benchmark.value,
                "judge_id": judge,
                "rounds": rounds,
                "elo_ratings": elo_ratings,
            },
            f,
            indent=2,
        )

    rprint(f"\n[dim]Results saved to: {results_file}[/dim]")


@app.command()
def leaderboard(
    evaluations_dir: Annotated[
        Path, typer.Option("--evaluations-dir", "-e", help="ModelEvaluationData directory")
    ] = Path("results/evaluations"),
    output: Annotated[Path, typer.Option("--output", "-o", help="Output file")] = Path(
        "leaderboard.html"
    ),
    format_type: Annotated[str, typer.Option("--format", "-f", help="Output format")] = "html",
    limit: Annotated[int | None, typer.Option("--limit", "-n", help="Max entries")] = None,
    registry_path: Annotated[
        Path | None, typer.Option("--registry", help="ELO registry path")
    ] = None,
) -> None:
    """Generate leaderboard from model evaluations and ELO registry."""
    from ukrqualbench.core.elo_registry import ELORegistry
    from ukrqualbench.core.schemas import ModelEvaluationData

    rprint(f"[yellow]Loading evaluations from:[/yellow] {evaluations_dir}")

    evaluations: dict[str, ModelEvaluationData] = {}
    eval_files = list(evaluations_dir.glob("*.json"))

    for eval_file in eval_files:
        try:
            with open(eval_file) as f:
                data = json.load(f)
                loaded_eval = ModelEvaluationData.from_dict(data)
                evaluations[loaded_eval.model_id] = loaded_eval
        except Exception as e:
            rprint(f"[yellow]Warning: Could not load {eval_file}: {e}[/yellow]")

    rprint(f"[green]Loaded {len(evaluations)} model evaluations[/green]")

    registry = ELORegistry(registry_path=registry_path)
    rprint(f"[green]Loaded ELO registry: {registry.model_count} models[/green]")

    table = Table(title="Leaderboard")
    table.add_column("Rank", style="dim", justify="right")
    table.add_column("Model", style="cyan")
    table.add_column("ELO", style="green", justify="right")
    table.add_column("MC Acc", justify="right")
    table.add_column("GEC F1", justify="right")
    table.add_column("Fertility", justify="right")
    table.add_column("Russisms", justify="right")
    table.add_column("Markers", justify="right")
    table.add_column("Status", justify="center")

    rankings = registry.get_rankings()
    leaderboard_data: list[dict[str, Any]] = []

    for rank, (model_id, elo) in enumerate(rankings[:limit] if limit else rankings, start=1):
        entry = registry.get_model(model_id)
        eval_data = evaluations.get(model_id)

        mc_acc = f"{eval_data.block_a.mc_accuracy:.1%}" if eval_data else "-"
        gec_f1 = f"{eval_data.block_a.gec_f1:.1%}" if eval_data else "-"
        fertility = f"{eval_data.block_v.fertility_rate:.2f}" if eval_data else "-"
        russisms = f"{eval_data.block_v.russism_rate:.1f}" if eval_data else "-"
        markers = f"{eval_data.block_v.positive_markers:.1f}" if eval_data else "-"
        status = (
            "[dim]provisional[/dim]" if entry and entry.is_provisional else "[green]stable[/green]"
        )

        table.add_row(
            str(rank), model_id, f"{elo:.1f}", mc_acc, gec_f1, fertility, russisms, markers, status
        )
        leaderboard_data.append(
            {
                "rank": rank,
                "model_id": model_id,
                "elo_rating": round(elo, 1),
                "mc_accuracy": eval_data.block_a.mc_accuracy if eval_data else None,
                "gec_f1": eval_data.block_a.gec_f1 if eval_data else None,
                "fertility_rate": eval_data.block_v.fertility_rate if eval_data else None,
                "russism_rate": eval_data.block_v.russism_rate if eval_data else None,
                "positive_markers": eval_data.block_v.positive_markers if eval_data else None,
                "provisional": entry.is_provisional if entry else True,
            }
        )

    console.print(table)

    if format_type == "json":
        content = json.dumps({"leaderboard": leaderboard_data}, indent=2)
        output = output.with_suffix(".json")
    elif format_type == "csv":
        import csv as csv_module
        import io

        buf = io.StringIO()
        if leaderboard_data:
            writer = csv_module.DictWriter(buf, fieldnames=leaderboard_data[0].keys())
            writer.writeheader()
            writer.writerows(leaderboard_data)
        content = buf.getvalue()
        output = output.with_suffix(".csv")
    elif format_type == "html":
        content = _generate_simple_html_leaderboard(leaderboard_data)
        output = output.with_suffix(".html")
    else:
        content = _generate_markdown_table(leaderboard_data)
        output = output.with_suffix(".md")

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        f.write(content)

    rprint(f"\n[green]Leaderboard saved to: {output}[/green]")


def _generate_simple_html_leaderboard(data: list[dict[str, Any]]) -> str:
    rows = ""
    for entry in data:
        rows += (
            f"""<tr>
            <td>{entry["rank"]}</td>
            <td>{entry["model_id"]}</td>
            <td>{entry["elo_rating"]}</td>
            <td>{entry["mc_accuracy"]:.1%}</td>
            <td>{entry["gec_f1"]:.1%}</td>
            <td>{entry["fertility_rate"]:.2f}</td>
            <td>{entry["russism_rate"]:.1f}</td>
            <td>{entry["positive_markers"]:.1f}</td>
        </tr>"""
            if entry["mc_accuracy"]
            else f"""<tr>
            <td>{entry["rank"]}</td>
            <td>{entry["model_id"]}</td>
            <td>{entry["elo_rating"]}</td>
            <td>-</td><td>-</td><td>-</td><td>-</td><td>-</td>
        </tr>"""
        )
    return f"""<!DOCTYPE html>
<html><head><title>UkrQualBench Leaderboard</title>
<style>table {{border-collapse: collapse; width: 100%;}} th,td {{border: 1px solid #ddd; padding: 8px; text-align: left;}}</style>
</head><body>
<h1>UkrQualBench Leaderboard</h1>
<table><tr><th>Rank</th><th>Model</th><th>ELO</th><th>MC Acc</th><th>GEC F1</th><th>Fertility</th><th>Russisms</th><th>Markers</th></tr>
{rows}</table></body></html>"""


def _generate_markdown_table(data: list[dict[str, Any]]) -> str:
    lines = [
        "| Rank | Model | ELO | MC Acc | GEC F1 | Fertility | Russisms | Markers |",
        "|------|-------|-----|--------|--------|-----------|----------|---------|",
    ]
    for entry in data:
        if entry["mc_accuracy"]:
            lines.append(
                f"| {entry['rank']} | {entry['model_id']} | {entry['elo_rating']} | {entry['mc_accuracy']:.1%} | {entry['gec_f1']:.1%} | {entry['fertility_rate']:.2f} | {entry['russism_rate']:.1f} | {entry['positive_markers']:.1f} |"
            )
        else:
            lines.append(
                f"| {entry['rank']} | {entry['model_id']} | {entry['elo_rating']} | - | - | - | - | - |"
            )
    return "\n".join(lines)


@app.command()
def info() -> None:
    """Show configuration and environment information."""
    config = Config()

    rprint(Panel.fit(f"[bold blue]UkrQualBench[/bold blue] v{__version__}", title="Version"))

    table = Table(title="Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Benchmark Version", config.benchmark_version.value)
    table.add_row("Default Judge", config.default_judge)
    table.add_row("ELO Initial Rating", str(config.elo_initial_rating))
    table.add_row("ELO K-Factor", str(config.elo_k_factor))
    table.add_row("Max Concurrent Requests", str(config.max_concurrent_requests))
    table.add_row("Request Timeout", f"{config.request_timeout}s")
    table.add_row("Temperature", str(config.temperature))
    table.add_row("Data Directory", str(config.data_dir))
    table.add_row("Results Directory", str(config.results_dir))

    console.print(table)

    api_table = Table(title="API Keys")
    api_table.add_column("Provider", style="cyan")
    api_table.add_column("Status", style="green")

    api_table.add_row(
        "OpenAI",
        "[green]\u2713 Configured[/green]"
        if config.openai_api_key
        else "[red]\u2717 Not set[/red]",
    )
    api_table.add_row(
        "Anthropic",
        "[green]\u2713 Configured[/green]"
        if config.anthropic_api_key
        else "[red]\u2717 Not set[/red]",
    )
    api_table.add_row(
        "Google",
        "[green]\u2713 Configured[/green]"
        if config.google_api_key
        else "[red]\u2717 Not set[/red]",
    )
    api_table.add_row(
        "Nebius",
        "[green]\u2713 Configured[/green]"
        if config.nebius_api_key
        else "[red]\u2717 Not set[/red]",
    )
    local_url = config.local_base_url or "Not configured"
    api_table.add_row("Local (LM Studio/etc)", f"[dim]{local_url}[/dim]")

    console.print(api_table)

    size_table = Table(title="Benchmark Sizes")
    size_table.add_column("Version", style="cyan")
    size_table.add_column("Block A", style="green")
    size_table.add_column("Block B", style="green")
    size_table.add_column("Est. Time", style="yellow")

    size_table.add_row("lite", "200", "100", "~30 min")
    size_table.add_row("base", "550", "250", "~2 hr")
    size_table.add_row("large", "1100", "450", "~5 hr")

    console.print(size_table)


@app.command()
def elo(
    action: Annotated[
        str,
        typer.Argument(help="Action: show, reset, export, history"),
    ] = "show",
    model: Annotated[str | None, typer.Option("--model", "-m", help="Filter by model")] = None,
    limit: Annotated[int, typer.Option("--limit", "-n", help="Limit results")] = 20,
    output: Annotated[Path | None, typer.Option("--output", "-o", help="Export file")] = None,
) -> None:
    """View and manage persistent ELO registry."""
    from ukrqualbench.core.elo_registry import ELORegistry

    config = Config()
    registry_path = config.data_dir / "elo_registry.json"

    if not registry_path.exists() and action != "reset":
        rprint(f"[yellow]No ELO registry found at {registry_path}[/yellow]")
        rprint("[dim]Run 'ukrqualbench compare' with --persistent to create one.[/dim]")
        raise typer.Exit(0)

    registry = ELORegistry(
        registry_path=registry_path,
        initial_rating=config.elo_initial_rating,
        k_factor=config.elo_k_factor,
    )

    if action == "show":
        if registry.model_count == 0:
            rprint("[yellow]Registry is empty.[/yellow]")
            raise typer.Exit(0)

        leaderboard = registry.get_leaderboard()

        table = Table(title=f"ELO Leaderboard ({registry.model_count} models)")
        table.add_column("#", style="dim")
        table.add_column("Model", style="cyan")
        table.add_column("ELO", style="green", justify="right")
        table.add_column("W/L/T", style="yellow", justify="center")
        table.add_column("Win%", style="blue", justify="right")
        table.add_column("Status", style="dim")

        for entry in leaderboard[:limit]:
            status = "[yellow]provisional[/yellow]" if entry["provisional"] else ""
            wlt = f"{entry['wins']}/{entry['losses']}/{entry['ties']}"
            win_pct = f"{entry['win_rate']:.1%}" if entry["games"] > 0 else "-"
            table.add_row(
                str(entry["rank"]),
                entry["model_id"],
                f"{entry['rating']:.0f}",
                wlt,
                win_pct,
                status,
            )

        console.print(table)
        rprint(f"\n[dim]Total comparisons: {registry.comparison_count}[/dim]")
        rprint(f"[dim]Registry: {registry_path}[/dim]")

    elif action == "history":
        from ukrqualbench.core.elo_registry import ComparisonLogEntry

        entries: list[ComparisonLogEntry]
        if model:
            entries = registry.get_model_history(model)
            title = f"History for {model}"
        else:
            entries = registry.get_recent_comparisons(limit)
            title = f"Recent Comparisons (last {limit})"

        if not entries:
            rprint("[yellow]No comparison history found.[/yellow]")
            raise typer.Exit(0)

        table = Table(title=title)
        table.add_column("Time", style="dim")
        table.add_column("Model A", style="cyan")
        table.add_column("Model B", style="cyan")
        table.add_column("Winner", style="green")
        table.add_column("Rating Change", style="yellow")

        for log in entries[-limit:]:
            ts = log.timestamp[:16].replace("T", " ")
            delta_a = log.new_rating_a - log.old_rating_a
            delta_b = log.new_rating_b - log.old_rating_b

            if log.winner == "A":
                winner = log.model_a
                change = f"+{delta_a:.0f} / {delta_b:.0f}"
            elif log.winner == "B":
                winner = log.model_b
                change = f"{delta_a:.0f} / +{abs(delta_b):.0f}"
            else:
                winner = "tie"
                change = f"{delta_a:+.0f} / {delta_b:+.0f}"

            table.add_row(ts, log.model_a, log.model_b, winner, change)

        console.print(table)

    elif action == "export":
        export_path = output or Path("elo_export.json")
        data = {
            "models": registry.get_leaderboard(),
            "metadata": registry.metadata.to_dict(),
        }
        with open(export_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        rprint(f"[green]Exported to: {export_path}[/green]")

    elif action == "reset":
        if registry.model_count > 0:
            confirm = typer.confirm(
                f"This will delete {registry.model_count} models and {registry.comparison_count} comparisons. Continue?"
            )
            if not confirm:
                raise typer.Abort()
        registry.reset()
        registry.save()
        rprint("[green]Registry reset.[/green]")

    else:
        rprint(f"[red]Unknown action: {action}[/red]")
        rprint("[dim]Valid actions: show, history, export, reset[/dim]")
        raise typer.Exit(1)


@app.command()
def benchmark(
    version: Annotated[
        str,
        typer.Option(
            "--version",
            "-v",
            help="Benchmark version: lite, base, or large",
        ),
    ] = "base",
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output file path (defaults to data/benchmarks/{version}.json)",
        ),
    ] = None,
    show_stats: Annotated[
        bool,
        typer.Option(
            "--stats",
            "-s",
            help="Show benchmark statistics",
        ),
    ] = False,
) -> None:
    """Assemble benchmark dataset from available sources."""
    from ukrqualbench.datasets import BENCHMARK_SPECS, BenchmarkAssembler

    if version not in BENCHMARK_SPECS:
        rprint(f"[red]Invalid version: {version}. Must be lite, base, or large.[/red]")
        raise typer.Exit(1)

    config = Config()
    assembler = BenchmarkAssembler(
        data_dir=config.data_dir,
        hf_token=config.huggingface_token,
    )

    rprint(f"[blue]Assembling {version} benchmark...[/blue]")

    benchmark_data = assembler.assemble(version)  # type: ignore[arg-type]

    if show_stats:
        stats_table = Table(title=f"Benchmark Statistics ({version})")
        stats_table.add_column("Component", style="cyan")
        stats_table.add_column("Count", style="green")

        stats_table.add_row("Block A - MC Tasks", str(len(benchmark_data.block_a.mc_tasks)))
        stats_table.add_row("Block A - GEC Tasks", str(len(benchmark_data.block_a.gec_tasks)))
        stats_table.add_row(
            "Block A - Translation", str(len(benchmark_data.block_a.translation_tasks))
        )
        stats_table.add_row(
            "Block A - False Positive", str(len(benchmark_data.block_a.false_positive_tasks))
        )
        stats_table.add_row(
            "Block A - Positive Markers", str(len(benchmark_data.block_a.positive_marker_tasks))
        )
        stats_table.add_row(
            "[bold]Block A Total[/bold]", f"[bold]{benchmark_data.block_a.total}[/bold]"
        )
        stats_table.add_row("", "")
        stats_table.add_row(
            "Block B - Generation", str(len(benchmark_data.block_b.generation_tasks))
        )
        stats_table.add_row(
            "Block B - Adversarial", str(len(benchmark_data.block_b.adversarial_tasks))
        )
        stats_table.add_row(
            "Block B - Long Context", str(len(benchmark_data.block_b.long_context_tasks))
        )
        stats_table.add_row(
            "[bold]Block B Total[/bold]", f"[bold]{benchmark_data.block_b.total}[/bold]"
        )
        stats_table.add_row("", "")
        stats_table.add_row(
            "[bold cyan]Grand Total[/bold cyan]",
            f"[bold cyan]{benchmark_data.total_tasks}[/bold cyan]",
        )

        console.print(stats_table)

        if benchmark_data.metadata:
            rprint(f"\n[dim]Dataset hash: {benchmark_data.metadata.dataset_hash[:16]}...[/dim]")
            rprint(f"[dim]Sources: {', '.join(benchmark_data.metadata.sources)}[/dim]")

    output_path = output or config.data_dir / "benchmarks" / f"{version}.json"

    from ukrqualbench.datasets import BenchmarkLoader

    loader = BenchmarkLoader(config.data_dir)
    loader.save_benchmark(benchmark_data, output_path)

    rprint(f"\n[green]Benchmark saved to: {output_path}[/green]")


if __name__ == "__main__":
    app()
