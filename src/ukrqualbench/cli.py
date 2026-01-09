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
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from ukrqualbench import __version__
from ukrqualbench.core.config import BenchmarkVersion, Config

if TYPE_CHECKING:
    from ukrqualbench.core.schemas import EvaluationResultData

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
        create_nebius_client,
        create_ollama_client,
        create_openai_client,
    )

    config = config or Config()
    model_lower = model_id.lower()

    if model_lower.startswith(("gpt-", "o1", "o3")):
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
    elif "/" in model_id:
        return create_nebius_client(
            model_id=model_id,
            api_key=config.nebius_api_key,
            temperature=config.temperature,
            max_retries=config.max_retries,
        )
    else:
        return create_ollama_client(
            model_id=model_id,
            base_url=config.ollama_base_url,
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

    async def run_calibration() -> Any:
        return await calibrator.calibrate(calibration_tasks)

    rprint(f"[cyan]Running calibration on {judge}... (this may take several minutes)[/cyan]")
    result = asyncio.run(run_calibration())

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
    judge: Annotated[
        str, typer.Option("--judge", "-j", help="Judge model")
    ] = "claude-3-5-haiku-latest",
    output: Annotated[Path, typer.Option("--output", "-o", help="Output directory")] = Path(
        "results"
    ),
    max_cost: Annotated[float | None, typer.Option("--max-cost", help="Max cost USD")] = None,
    resume: Annotated[bool, typer.Option("--resume", "-r", help="Resume from checkpoint")] = True,
) -> None:
    """Evaluate a single model on the benchmark."""
    from ukrqualbench.core.evaluator import Evaluator

    config = Config()
    config.benchmark_version = benchmark
    if max_cost is not None:
        config.max_cost_usd = max_cost

    output.mkdir(parents=True, exist_ok=True)

    rprint(
        Panel.fit(
            f"[bold]Evaluating Model[/bold]\n\nModel: [cyan]{model}[/cyan]\nBenchmark: [yellow]{benchmark.value}[/yellow]\nJudge: [dim]{judge}[/dim]",
            title="UkrQualBench Evaluation",
        )
    )

    evaluator = Evaluator(config=config)

    async def run_evaluation() -> Any:
        return await evaluator.evaluate_model(model_id=model, judge_id=judge, resume=resume)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"[cyan]Evaluating {model}...", total=None)
        result = asyncio.run(run_evaluation())
        progress.update(task, completed=True)

    _display_evaluation_results(result)

    result_file = output / f"{model.replace('/', '_')}_evaluation.json"
    with open(result_file, "w") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)

    rprint(f"\n[dim]Results saved to: {result_file}[/dim]")


def _display_evaluation_results(result: EvaluationResultData) -> None:
    table = Table(title=f"Evaluation Results: {result.model_id}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    scores = result.scores

    table.add_row("[bold]Block A (Calibration)[/bold]", "")
    table.add_row("  MC Accuracy", f"{scores.block_a.mc_accuracy:.1%}")
    table.add_row("  GEC F1", f"{scores.block_a.gec_f1:.1%}")
    table.add_row("  Translation COMET", f"{scores.block_a.translation_comet:.3f}")

    table.add_row("[bold]Block B (Generation)[/bold]", "")
    table.add_row("  Generation ELO", f"{scores.block_b.generation_elo:.0f}")
    table.add_row("  Adversarial ELO", f"{scores.block_b.adversarial_elo:.0f}")
    table.add_row("  Long Context ELO", f"{scores.block_b.long_context_elo:.0f}")

    table.add_row("[bold]Block V (Metrics)[/bold]", "")
    table.add_row("  Fertility Rate", f"{scores.block_v.fertility_rate:.2f}")
    table.add_row("  Positive Markers", f"{scores.block_v.positive_markers:.1f}")
    table.add_row("  Russism Rate", f"{scores.block_v.russism_rate:.2f}")
    table.add_row("  Anglicism Rate", f"{scores.block_v.anglicism_rate:.2f}")

    table.add_row("[bold]Overall[/bold]", "")
    table.add_row("  ELO Rating", f"[bold]{scores.elo_rating:.0f}[/bold]")

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
) -> None:
    """Compare multiple models using pairwise evaluation."""
    import math

    from ukrqualbench.core.evaluator import Evaluator

    config = Config()
    config.benchmark_version = benchmark
    if max_cost is not None:
        config.max_cost_usd = max_cost

    model_list = [m.strip() for m in models.split(",")]
    n_models = len(model_list)

    if rounds is None:
        rounds = math.ceil(math.log2(max(n_models, 2))) + 2

    output.mkdir(parents=True, exist_ok=True)

    rprint(
        Panel.fit(
            f"[bold]Comparing Models[/bold]\n\nModels: [cyan]{n_models}[/cyan]\n"
            + "\n".join(f"  \u2022 {m}" for m in model_list)
            + f"\n\nBenchmark: [yellow]{benchmark.value}[/yellow]\nRounds: [yellow]{rounds}[/yellow]",
            title="UkrQualBench Comparison",
        )
    )

    evaluator = Evaluator(config=config)

    async def run_comparison() -> Any:
        return await evaluator.compare_models(model_ids=model_list, judge_id=judge, rounds=rounds)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"[cyan]Running {rounds} rounds of comparison...", total=rounds * n_models
        )
        results = asyncio.run(run_comparison())
        progress.update(task, completed=True)

    from ukrqualbench.reports import create_leaderboard

    leaderboard = create_leaderboard(
        results=results, benchmark_version=benchmark.value, judge_id=judge
    )

    rprint("\n" + leaderboard.to_table(format="unicode"))

    results_file = output / "comparison_results.json"
    with open(results_file, "w") as f:
        f.write(leaderboard.to_json())

    rprint(f"\n[dim]Results saved to: {results_file}[/dim]")


@app.command()
def leaderboard(
    results_dir: Annotated[
        Path, typer.Option("--results-dir", "-r", help="Results directory")
    ] = Path("results"),
    output: Annotated[Path, typer.Option("--output", "-o", help="Output file")] = Path(
        "leaderboard.html"
    ),
    format_type: Annotated[str, typer.Option("--format", "-f", help="Output format")] = "html",
    limit: Annotated[int | None, typer.Option("--limit", "-n", help="Max entries")] = None,
) -> None:
    """Generate leaderboard from evaluation results."""
    from ukrqualbench.core.schemas import EvaluationResultData
    from ukrqualbench.reports import LeaderboardGenerator, generate_leaderboard_html

    rprint(f"[yellow]Generating leaderboard from:[/yellow] {results_dir}")

    results: list[EvaluationResultData] = []
    result_files = list(results_dir.glob("*_evaluation.json"))

    if not result_files:
        rprint("[red]No evaluation results found in the specified directory.[/red]")
        raise typer.Exit(1)

    for result_file in result_files:
        try:
            with open(result_file) as f:
                data = json.load(f)
                results.append(EvaluationResultData.from_dict(data))
        except Exception as e:
            rprint(f"[yellow]Warning: Could not load {result_file}: {e}[/yellow]")

    if not results:
        rprint("[red]No valid evaluation results found.[/red]")
        raise typer.Exit(1)

    rprint(f"[green]Loaded {len(results)} evaluation results[/green]")

    generator = LeaderboardGenerator()
    for result in results:
        generator.add_result(result)
    generator.finalize()

    if format_type == "html":
        content = generate_leaderboard_html(generator)
        output = output.with_suffix(".html")
    elif format_type == "json":
        content = generator.to_json()
        output = output.with_suffix(".json")
    elif format_type == "csv":
        content = generator.to_csv(include_details=True)
        output = output.with_suffix(".csv")
    elif format_type == "markdown":
        content = generator.to_table(format="markdown")
        output = output.with_suffix(".md")
    else:
        rprint(f"[red]Unknown format: {format_type}[/red]")
        raise typer.Exit(1)

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        f.write(content)

    rprint(f"[green]Leaderboard saved to: {output}[/green]")
    rprint("\n" + generator.to_table(format="unicode"))


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
    api_table.add_row("Ollama", f"[dim]{config.ollama_base_url}[/dim]")

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


if __name__ == "__main__":
    app()
