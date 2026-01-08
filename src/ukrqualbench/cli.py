"""CLI entry point for UkrQualBench.

Commands:
- calibrate: Calibrate a judge model
- evaluate: Evaluate a single model
- compare: Compare multiple models
- leaderboard: Generate leaderboard from results
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ukrqualbench import __version__
from ukrqualbench.core.config import BenchmarkVersion, Config

app = typer.Typer(
    name="ukrqualbench",
    help="Benchmark for evaluating Ukrainian language quality in LLMs",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


def version_callback(value: bool) -> None:
    """Print version and exit."""
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
    pass


@app.command()
def calibrate(
    judge: Annotated[
        str,
        typer.Option(
            "--judge",
            "-j",
            help="Judge model to calibrate (e.g., claude-3-5-haiku-latest)",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output directory for calibration results",
        ),
    ] = Path("results/calibration"),
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show detailed output"),
    ] = False,
) -> None:
    """Calibrate a judge model before using it for evaluation.

    The judge will be tested on:
    - Multiple choice agreement (threshold: >85%)
    - GEC F1 score (threshold: >80%)
    - Russism detection F1 (threshold: >85%)
    - False positive rate (threshold: <15%)
    - Pairwise consistency (threshold: >90%)

    Final acceptance threshold: score > 0.80
    """
    rprint(f"[yellow]Calibrating judge:[/yellow] {judge}")
    rprint(f"[dim]Output directory:[/dim] {output}")

    # TODO: Implement calibration logic
    rprint("[red]Calibration not yet implemented[/red]")
    raise typer.Exit(1)


@app.command()
def evaluate(
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Model to evaluate"),
    ],
    benchmark: Annotated[
        BenchmarkVersion,
        typer.Option(
            "--benchmark",
            "-b",
            help="Benchmark version (lite ~30min, base ~2hr, large ~5hr)",
        ),
    ] = BenchmarkVersion.BASE,
    judge: Annotated[
        str,
        typer.Option("--judge", "-j", help="Judge model for pairwise comparisons"),
    ] = "claude-3-5-haiku-latest",
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output directory for results"),
    ] = Path("results"),
    max_cost: Annotated[
        float | None,
        typer.Option("--max-cost", help="Maximum cost in USD (stops if exceeded)"),
    ] = None,
    resume: Annotated[
        bool,
        typer.Option("--resume", "-r", help="Resume from last checkpoint"),
    ] = True,
) -> None:
    """Evaluate a single model on the benchmark.

    Runs all three blocks:
    - Block A: Calibration tests (MC, GEC, translation, false positives)
    - Block B: Generation tests (pairwise evaluation)
    - Block V: Automatic metrics (fertility, markers, russisms)
    """
    rprint(f"[yellow]Evaluating model:[/yellow] {model}")
    rprint(f"[dim]Benchmark:[/dim] {benchmark.value}")
    rprint(f"[dim]Judge:[/dim] {judge}")
    rprint(f"[dim]Output:[/dim] {output}")
    if max_cost:
        rprint(f"[dim]Max cost:[/dim] ${max_cost:.2f}")

    # TODO: Implement evaluation logic
    rprint("[red]Evaluation not yet implemented[/red]")
    raise typer.Exit(1)


@app.command()
def compare(
    models: Annotated[
        str,
        typer.Option(
            "--models",
            "-m",
            help="Comma-separated list of models to compare",
        ),
    ],
    benchmark: Annotated[
        BenchmarkVersion,
        typer.Option("--benchmark", "-b", help="Benchmark version"),
    ] = BenchmarkVersion.BASE,
    judge: Annotated[
        str,
        typer.Option("--judge", "-j", help="Judge model for comparisons"),
    ] = "claude-3-5-haiku-latest",
    rounds: Annotated[
        int | None,
        typer.Option("--rounds", help="Number of tournament rounds (default: auto)"),
    ] = None,
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output directory"),
    ] = Path("results"),
    max_cost: Annotated[
        float | None,
        typer.Option("--max-cost", help="Maximum total cost in USD"),
    ] = None,
) -> None:
    """Compare multiple models using pairwise evaluation.

    Uses Swiss-system tournament (not all pairs) for efficiency.
    Default rounds: ceil(log2(n)) + 2 where n is number of models.
    """
    model_list = [m.strip() for m in models.split(",")]
    rprint(f"[yellow]Comparing {len(model_list)} models:[/yellow]")
    for m in model_list:
        rprint(f"  • {m}")
    rprint(f"[dim]Benchmark:[/dim] {benchmark.value}")
    rprint(f"[dim]Judge:[/dim] {judge}")
    if rounds:
        rprint(f"[dim]Rounds:[/dim] {rounds}")
    if max_cost:
        rprint(f"[dim]Max cost:[/dim] ${max_cost:.2f}")

    # TODO: Implement comparison logic
    rprint("[red]Comparison not yet implemented[/red]")
    raise typer.Exit(1)


@app.command()
def leaderboard(
    results_dir: Annotated[
        Path,
        typer.Option(
            "--results-dir",
            "-r",
            help="Directory containing evaluation results",
        ),
    ] = Path("results"),
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output file (HTML or JSON)"),
    ] = Path("leaderboard.html"),
    format_type: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format (html or json)"),
    ] = "html",
) -> None:
    """Generate leaderboard from evaluation results."""
    rprint(f"[yellow]Generating leaderboard from:[/yellow] {results_dir}")
    rprint(f"[dim]Output:[/dim] {output}")
    rprint(f"[dim]Format:[/dim] {format_type}")

    # TODO: Implement leaderboard generation
    rprint("[red]Leaderboard generation not yet implemented[/red]")
    raise typer.Exit(1)


@app.command()
def info() -> None:
    """Show configuration and environment information."""
    config = Config()

    # Version info
    rprint(Panel.fit(
        f"[bold blue]UkrQualBench[/bold blue] v{__version__}",
        title="Version",
    ))

    # Configuration table
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

    # API keys status
    api_table = Table(title="API Keys")
    api_table.add_column("Provider", style="cyan")
    api_table.add_column("Status", style="green")

    api_table.add_row(
        "OpenAI",
        "[green]✓ Configured[/green]" if config.openai_api_key else "[red]✗ Not set[/red]",
    )
    api_table.add_row(
        "Anthropic",
        "[green]✓ Configured[/green]" if config.anthropic_api_key else "[red]✗ Not set[/red]",
    )
    api_table.add_row(
        "Google",
        "[green]✓ Configured[/green]" if config.google_api_key else "[red]✗ Not set[/red]",
    )
    api_table.add_row("Ollama", f"[dim]{config.ollama_base_url}[/dim]")

    console.print(api_table)

    # Benchmark sizes
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
