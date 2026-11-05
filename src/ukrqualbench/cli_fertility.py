"""``ukrqualbench fertility`` commands: calibrate, measure, show, apply."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from ukrqualbench.core.config import Config
from ukrqualbench.tokenization.corpus import load_reference_corpus
from ukrqualbench.tokenization.registry import (
    FertilityRegistry,
    default_registry_path,
    fertility_block_v_fields,
)
from ukrqualbench.tokenization.specs import MODEL_TOKENIZERS

if TYPE_CHECKING:
    from collections.abc import Callable

    from ukrqualbench.tokenization.counters import TokenCounter
    from ukrqualbench.tokenization.specs import TokenizerSpec

fertility_app = typer.Typer(
    help="Native-tokenizer fertility rate on the UA-GEC reference corpus.",
    no_args_is_help=True,
)
console = Console()

DEFAULT_EVALUATIONS_DIR = Path("results/evaluations")

RegistryOption = Annotated[
    Path | None,
    typer.Option("--registry", help="Registry path (default: <data_dir>/fertility/registry.json)"),
]
EvaluationsOption = Annotated[
    Path, typer.Option("--evaluations-dir", "-e", help="ModelEvaluationData directory")
]


def _load_registry(registry_path: Path | None, config: Config) -> FertilityRegistry:
    corpus = load_reference_corpus()
    path = registry_path or default_registry_path(config)
    return FertilityRegistry.load(path, expected_corpus_sha256=corpus.sha256)


def _evaluation_model_ids(evaluations_dir: Path) -> dict[str, Path]:
    model_ids: dict[str, Path] = {}
    for eval_file in sorted(evaluations_dir.glob("*.json")):
        with open(eval_file, encoding="utf-8") as f:
            model_ids[json.load(f)["model_id"]] = eval_file
    return model_ids


@fertility_app.command("measure")
def measure(
    models: Annotated[
        list[str] | None, typer.Argument(help="Model ids to measure (see `fertility show`)")
    ] = None,
    all_models: Annotated[
        bool, typer.Option("--all", help="Measure every model with a tokenizer spec")
    ] = False,
    force: Annotated[bool, typer.Option("--force", help="Re-measure valid entries")] = False,
    registry_path: RegistryOption = None,
) -> None:
    """Count the reference corpus with each model's native tokenizer."""
    from ukrqualbench.tokenization.counters import ModelNotFoundError
    from ukrqualbench.tokenization.measure import UnknownModelError, measure_model

    targets = sorted(MODEL_TOKENIZERS) if all_models else list(models or [])
    if not targets:
        rprint("[red]Name model ids or pass --all[/red]")
        raise typer.Exit(2)

    config = Config()
    corpus = load_reference_corpus()
    registry = _load_registry(registry_path, config)
    rprint(
        f"[cyan]Corpus {corpus.corpus_id}: {len(corpus.texts)} docs, "
        f"{corpus.words} words, sha256 {corpus.sha256[:16]}[/cyan]"
    )

    failures = 0
    for model_id in targets:
        if not force and registry.get(model_id) is not None:
            rprint(f"[dim]{model_id}: valid entry exists, skipped (--force to re-measure)[/dim]")
            continue
        try:
            entry = asyncio.run(measure_model(model_id, config, corpus=corpus))
        except ModelNotFoundError as e:
            rprint(
                f"[yellow]{model_id}: API does not know the model and there is no verified "
                f"fallback; fertility stays empty — {e}[/yellow]"
            )
            continue
        except UnknownModelError as e:
            rprint(f"[red]{model_id}: {e}[/red]")
            failures += 1
            continue
        except Exception as e:
            rprint(f"[red]{model_id}: {type(e).__name__}: {e}[/red]")
            failures += 1
            continue
        registry.put(entry)
        registry.save(corpus)
        rprint(
            f"[green]{model_id}: tokens={entry.tokens} FR={entry.fertility_rate:.4f} "
            f"chars/token={entry.chars_per_token:.3f} ({entry.method})[/green]"
        )

    if failures:
        raise typer.Exit(1)


CALIBRATION_PROVIDERS = ("openai", "gemini", "anthropic")


async def _calibrate(config: Config, providers: list[str], info: bool) -> bool:
    """Run the calibration checks of plan phase 2; return True if all passed."""
    from ukrqualbench.tokenization.counters import make_counter

    opened: list[TokenCounter] = []

    def make(spec: TokenizerSpec) -> TokenCounter:
        counter = make_counter(spec, config)
        opened.append(counter)
        return counter

    try:
        return await _run_checks(make, providers, info)
    finally:
        for counter in opened:
            await counter.aclose()


async def _run_checks(
    make: Callable[[TokenizerSpec], TokenCounter], providers: list[str], info: bool
) -> bool:
    from ukrqualbench.tokenization import calibrate as cal
    from ukrqualbench.tokenization.specs import GEMMA3, GEMMA4_31B, O200K_BASE

    texts = load_reference_corpus().texts
    o200k = make(O200K_BASE)
    passed = True

    def api(model_id: str) -> TokenCounter:
        return make(MODEL_TOKENIZERS[model_id])

    def report_exact(result: cal.ExactMatchResult) -> None:
        nonlocal passed
        passed = passed and result.passed
        verdict = "[green]PASS[/green]" if result.passed else "[red]FAIL[/red]"
        detail = result.error or (
            f"{result.docs} docs, {result.mismatches} mismatches, overhead O={result.overhead}, "
            f"api-local diffs {result.diff_histogram or '{}'}"
        )
        rprint(f"{verdict} {result.label} vs {result.reference}: {detail}")

    if "openai" in providers:
        report_exact(await cal.exact_match("openai gpt-5-nano", api("gpt-5-nano"), o200k, texts))
        if info:
            for model_id in ("gpt-5.2", "gpt-5.5", "gpt-5.6-sol", "gpt-6-astra", "gpt-6.1-sol"):
                share = await cal.agreement_share(api(model_id), o200k, texts)
                rprint(f"[dim]info: {model_id} equals o200k_base on {share:.0%} of docs[/dim]")

    if "gemini" in providers:
        gemma3 = make(GEMMA3)
        report_exact(
            await cal.exact_match(
                "gemini gemini-3-flash-preview", api("gemini-3-flash-preview"), gemma3, texts
            )
        )
        if info:
            gemma4 = make(GEMMA4_31B)
            for model_id in ("gemini-3.5-flash", "gemini-3.8-flash"):
                share = await cal.agreement_share(api(model_id), gemma4, texts)
                rprint(f"[dim]info: {model_id} equals Gemma 4 on {share:.0%} of docs[/dim]")

    if "anthropic" in providers:
        # One model per measured Claude tokenizer variant.
        for model_id in ("claude-opus-5", "claude-opus-4-7", "claude-haiku-4-5"):
            split = await cal.split_additivity(f"anthropic {model_id}", api(model_id), o200k, texts)
            passed = passed and split.passed
            verdict = "[green]PASS[/green]" if split.passed else "[red]FAIL[/red]"
            detail = split.error or (
                f"max |n(A)+n(B)-n(doc)| = {split.max_gap} (tolerance {split.tolerance}) on "
                f"{len(split.gaps)} docs, overhead O={split.overhead}; gaps {split.gaps}; "
                f"o200k gaps {split.reference_gaps}"
            )
            rprint(f"{verdict} {split.label} split additivity: {detail}")

    return passed


@fertility_app.command("calibrate")
def calibrate(
    providers: Annotated[
        list[str] | None,
        typer.Option(
            "--provider", "-p", help="openai, gemini, anthropic (repeatable; default all)"
        ),
    ] = None,
    info: Annotated[
        bool,
        typer.Option("--info/--no-info", help="Also report agreement of other API models"),
    ] = True,
) -> None:
    """Check the API count scheme against local tokenizers before measuring."""
    selected = list(providers or CALIBRATION_PROVIDERS)
    unknown = sorted(set(selected) - set(CALIBRATION_PROVIDERS))
    if unknown:
        rprint(f"[red]Unknown provider(s): {', '.join(unknown)}[/red]")
        raise typer.Exit(2)
    try:
        passed = asyncio.run(_calibrate(Config(), selected, info))
    except Exception as e:
        rprint(f"[red]FAIL calibration aborted: {type(e).__name__}: {e}[/red]")
        raise typer.Exit(1) from e
    if not passed:
        raise typer.Exit(1)


@fertility_app.command("show")
def show(
    registry_path: RegistryOption = None,
    evaluations_dir: EvaluationsOption = DEFAULT_EVALUATIONS_DIR,
) -> None:
    """Print the registry and models that still lack a fertility value."""
    config = Config()
    registry = _load_registry(registry_path, config)

    table = Table(title=f"Fertility registry: {registry.path}")
    table.add_column("Model", style="cyan")
    table.add_column("Tokens", justify="right")
    table.add_column("FR", justify="right")
    table.add_column("Chars/token", justify="right")
    table.add_column("Method")
    table.add_column("Status")
    table.add_column("Note", style="dim")
    entries = registry.all_entries()
    for model_id in sorted(entries):
        entry = entries[model_id]
        status = "valid" if registry.is_valid(entry) else "[red]stale[/red]"
        table.add_row(
            model_id,
            str(entry.tokens),
            f"{entry.fertility_rate:.4f}",
            f"{entry.chars_per_token:.3f}",
            entry.method,
            status,
            entry.note,
        )
    console.print(table)

    if evaluations_dir.is_dir():
        evaluated = _evaluation_model_ids(evaluations_dir)
        no_spec = sorted(m for m in evaluated if m not in MODEL_TOKENIZERS)
        not_measured = sorted(
            m for m in evaluated if m in MODEL_TOKENIZERS and registry.get(m) is None
        )
        if no_spec:
            rprint(f"[red]Evaluated models without a tokenizer spec:[/red] {', '.join(no_spec)}")
        if not_measured:
            rprint(f"[yellow]Evaluated models not measured yet:[/yellow] {', '.join(not_measured)}")


@fertility_app.command("apply")
def apply(
    evaluations_dir: EvaluationsOption = DEFAULT_EVALUATIONS_DIR,
    registry_path: RegistryOption = None,
) -> None:
    """Write registry values into block_v.fertility_rate of every evaluation file.

    Models without a valid registry entry get ``null``. ``results/`` is not
    under git: back the directory up before running this command.
    """
    from ukrqualbench.core.schemas import ModelEvaluationData

    config = Config()
    registry = _load_registry(registry_path, config)

    for model_id, eval_file in _evaluation_model_ids(evaluations_dir).items():
        with open(eval_file, encoding="utf-8") as f:
            data = json.load(f)
        block_v = data.setdefault("block_v", {})
        old = block_v.get("fertility_rate")
        block_v.update(fertility_block_v_fields(registry, model_id))
        ModelEvaluationData.from_dict(data)
        with open(eval_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        new = block_v["fertility_rate"]
        rprint(f"{model_id}: {old} → {new}")
