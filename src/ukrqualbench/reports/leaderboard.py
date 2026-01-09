"""Leaderboard generation for model rankings.

Generates formatted leaderboards from evaluation results with:
- ELO-based rankings with confidence intervals
- Quality badges based on performance thresholds
- Multiple output formats (table, JSON, CSV)
- Breakdown by evaluation block (A, B, V)
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from ukrqualbench.core.schemas import Badge, EvaluationResultData, ModelScoreData

if TYPE_CHECKING:
    from collections.abc import Sequence


# ============================================================================
# Badge Assignment Thresholds
# ============================================================================

# Based on Section 3.3 of Technical Specification
BADGE_THRESHOLDS = {
    Badge.GOLD: {
        "min_elo": 1650,
        "max_russism_rate": 1.0,
        "min_positive_markers": 5.0,
        "max_fertility": 1.5,
    },
    Badge.SILVER: {
        "min_elo": 1550,
        "max_russism_rate": 3.0,
        "min_positive_markers": 3.0,
        "max_fertility": 1.8,
    },
    Badge.BRONZE: {
        "min_elo": 1450,
        "max_russism_rate": 5.0,
        "min_positive_markers": 1.0,
        "max_fertility": 2.0,
    },
    Badge.CAUTION: {
        "min_elo": 1350,
        "max_russism_rate": 10.0,
        "min_positive_markers": 0.0,
        "max_fertility": 2.5,
    },
}


def assign_badge(scores: ModelScoreData) -> Badge:
    """Assign quality badge based on scores.

    Args:
        scores: Model scores including ELO and block V metrics.

    Returns:
        Appropriate badge based on thresholds.
    """
    elo = scores.elo_rating
    russism_rate = scores.block_v.russism_rate
    positive_markers = scores.block_v.positive_markers
    fertility = scores.block_v.fertility_rate

    # Check badges in order of prestige
    for badge in [Badge.GOLD, Badge.SILVER, Badge.BRONZE, Badge.CAUTION]:
        thresholds = BADGE_THRESHOLDS[badge]
        if (
            elo >= thresholds["min_elo"]
            and russism_rate <= thresholds["max_russism_rate"]
            and positive_markers >= thresholds["min_positive_markers"]
            and fertility <= thresholds["max_fertility"]
        ):
            return badge

    return Badge.NOT_RECOMMENDED


# ============================================================================
# Leaderboard Entry
# ============================================================================


@dataclass
class LeaderboardEntry:
    """Single entry in the leaderboard."""

    rank: int
    model_id: str
    elo_rating: float
    badge: Badge

    # Block scores (optional, for detailed view)
    mc_accuracy: float | None = None
    gec_f1: float | None = None
    translation_comet: float | None = None

    generation_elo: float | None = None
    adversarial_elo: float | None = None
    long_context_elo: float | None = None

    fertility_rate: float | None = None
    positive_markers: float | None = None
    russism_rate: float | None = None
    anglicism_rate: float | None = None

    # Statistics
    wins: int = 0
    losses: int = 0
    ties: int = 0
    total_comparisons: int = 0

    # Confidence interval
    elo_ci_low: float | None = None
    elo_ci_high: float | None = None

    def to_dict(self, include_details: bool = True) -> dict[str, Any]:
        """Convert to dictionary.

        Args:
            include_details: Whether to include detailed block scores.

        Returns:
            Dictionary representation.
        """
        result: dict[str, Any] = {
            "rank": self.rank,
            "model_id": self.model_id,
            "elo_rating": round(self.elo_rating, 1),
            "badge": self.badge.value,
        }

        if include_details:
            result.update(
                {
                    "block_a": {
                        "mc_accuracy": self.mc_accuracy,
                        "gec_f1": self.gec_f1,
                        "translation_comet": self.translation_comet,
                    },
                    "block_b": {
                        "generation_elo": self.generation_elo,
                        "adversarial_elo": self.adversarial_elo,
                        "long_context_elo": self.long_context_elo,
                    },
                    "block_v": {
                        "fertility_rate": self.fertility_rate,
                        "positive_markers": self.positive_markers,
                        "russism_rate": self.russism_rate,
                        "anglicism_rate": self.anglicism_rate,
                    },
                    "statistics": {
                        "wins": self.wins,
                        "losses": self.losses,
                        "ties": self.ties,
                        "total_comparisons": self.total_comparisons,
                    },
                }
            )

            if self.elo_ci_low is not None and self.elo_ci_high is not None:
                result["confidence_interval"] = {
                    "low": round(self.elo_ci_low, 1),
                    "high": round(self.elo_ci_high, 1),
                }

        return result


# ============================================================================
# Leaderboard Generator
# ============================================================================


@dataclass
class LeaderboardGenerator:
    """Generator for leaderboards from evaluation results.

    Attributes:
        entries: List of leaderboard entries.
        metadata: Leaderboard metadata.
    """

    entries: list[LeaderboardEntry] = field(default_factory=list)
    metadata: LeaderboardMetadata | None = None

    def add_result(
        self,
        result: EvaluationResultData,
        wins: int = 0,
        losses: int = 0,
        ties: int = 0,
    ) -> None:
        """Add evaluation result to leaderboard.

        Args:
            result: Evaluation result for a model.
            wins: Number of pairwise wins.
            losses: Number of pairwise losses.
            ties: Number of ties.
        """
        scores = result.scores
        badge = assign_badge(scores)

        entry = LeaderboardEntry(
            rank=0,  # Will be set by finalize()
            model_id=result.model_id,
            elo_rating=scores.elo_rating,
            badge=badge,
            mc_accuracy=scores.block_a.mc_accuracy,
            gec_f1=scores.block_a.gec_f1,
            translation_comet=scores.block_a.translation_comet,
            generation_elo=scores.block_b.generation_elo,
            adversarial_elo=scores.block_b.adversarial_elo,
            long_context_elo=scores.block_b.long_context_elo,
            fertility_rate=scores.block_v.fertility_rate,
            positive_markers=scores.block_v.positive_markers,
            russism_rate=scores.block_v.russism_rate,
            anglicism_rate=scores.block_v.anglicism_rate,
            wins=wins,
            losses=losses,
            ties=ties,
            total_comparisons=result.comparisons_count,
        )
        self.entries.append(entry)

    def add_from_results(
        self,
        results: Sequence[EvaluationResultData],
        statistics: dict[str, dict[str, int]] | None = None,
    ) -> None:
        """Add multiple evaluation results.

        Args:
            results: Sequence of evaluation results.
            statistics: Optional dict mapping model_id to {wins, losses, ties}.
        """
        statistics = statistics or {}
        for result in results:
            stats = statistics.get(result.model_id, {})
            self.add_result(
                result,
                wins=stats.get("wins", 0),
                losses=stats.get("losses", 0),
                ties=stats.get("ties", 0),
            )

    def finalize(
        self,
        benchmark_version: str = "base",
        judge_id: str = "",
        total_prompts: int = 0,
    ) -> None:
        """Finalize leaderboard: sort and assign ranks.

        Args:
            benchmark_version: Version of benchmark used (lite/base/large).
            judge_id: Judge model ID.
            total_prompts: Total prompts evaluated.
        """
        # Sort by ELO rating (descending)
        self.entries.sort(key=lambda e: e.elo_rating, reverse=True)

        # Assign ranks
        for i, entry in enumerate(self.entries, start=1):
            entry.rank = i

        # Set metadata
        self.metadata = LeaderboardMetadata(
            benchmark_version=benchmark_version,
            judge_id=judge_id,
            total_models=len(self.entries),
            total_prompts=total_prompts,
            generated_at=datetime.now(),
        )

    def get_entries(
        self,
        limit: int | None = None,
        badge_filter: Badge | None = None,
    ) -> list[LeaderboardEntry]:
        """Get leaderboard entries with optional filtering.

        Args:
            limit: Maximum entries to return.
            badge_filter: Filter by specific badge.

        Returns:
            Filtered list of entries.
        """
        entries = self.entries

        if badge_filter is not None:
            entries = [e for e in entries if e.badge == badge_filter]

        if limit is not None:
            entries = entries[:limit]

        return entries

    def to_json(
        self,
        include_details: bool = True,
        indent: int = 2,
    ) -> str:
        """Export leaderboard to JSON.

        Args:
            include_details: Whether to include detailed block scores.
            indent: JSON indentation.

        Returns:
            JSON string.
        """
        data: dict[str, Any] = {
            "leaderboard": [
                entry.to_dict(include_details=include_details) for entry in self.entries
            ],
        }

        if self.metadata:
            data["metadata"] = self.metadata.to_dict()

        return json.dumps(data, indent=indent, default=str)

    def to_csv(
        self,
        include_details: bool = False,
    ) -> str:
        """Export leaderboard to CSV.

        Args:
            include_details: Whether to include detailed block scores.

        Returns:
            CSV string.
        """
        output = io.StringIO()

        if include_details:
            fieldnames = [
                "rank",
                "model_id",
                "elo_rating",
                "badge",
                "mc_accuracy",
                "gec_f1",
                "translation_comet",
                "generation_elo",
                "adversarial_elo",
                "long_context_elo",
                "fertility_rate",
                "positive_markers",
                "russism_rate",
                "anglicism_rate",
                "wins",
                "losses",
                "ties",
            ]
        else:
            fieldnames = ["rank", "model_id", "elo_rating", "badge"]

        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        for entry in self.entries:
            row: dict[str, Any] = {
                "rank": entry.rank,
                "model_id": entry.model_id,
                "elo_rating": round(entry.elo_rating, 1),
                "badge": entry.badge.value,
            }

            if include_details:
                row.update(
                    {
                        "mc_accuracy": entry.mc_accuracy,
                        "gec_f1": entry.gec_f1,
                        "translation_comet": entry.translation_comet,
                        "generation_elo": entry.generation_elo,
                        "adversarial_elo": entry.adversarial_elo,
                        "long_context_elo": entry.long_context_elo,
                        "fertility_rate": entry.fertility_rate,
                        "positive_markers": entry.positive_markers,
                        "russism_rate": entry.russism_rate,
                        "anglicism_rate": entry.anglicism_rate,
                        "wins": entry.wins,
                        "losses": entry.losses,
                        "ties": entry.ties,
                    }
                )

            writer.writerow(row)

        return output.getvalue()

    def to_table(
        self,
        format: Literal["markdown", "ascii", "unicode"] = "unicode",
        columns: list[str] | None = None,
    ) -> str:
        """Export leaderboard to formatted table.

        Args:
            format: Table format (markdown, ascii, unicode).
            columns: Columns to include (default: rank, model, elo, badge).

        Returns:
            Formatted table string.
        """
        if columns is None:
            columns = ["rank", "model_id", "elo_rating", "badge"]

        # Build header and rows
        header = [c.replace("_", " ").title() for c in columns]
        rows = []

        for entry in self.entries:
            row = []
            for col in columns:
                value = getattr(entry, col, "")
                if isinstance(value, float):
                    value = f"{value:.1f}"
                elif isinstance(value, Badge):
                    value = self._badge_symbol(value, format)
                row.append(str(value))
            rows.append(row)

        # Calculate column widths
        widths = [len(h) for h in header]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(cell))

        # Format based on style
        if format == "markdown":
            return self._format_markdown_table(header, rows, widths)
        elif format == "ascii":
            return self._format_ascii_table(header, rows, widths)
        else:  # unicode
            return self._format_unicode_table(header, rows, widths)

    def _badge_symbol(self, badge: Badge, format: str) -> str:
        """Get badge symbol for display."""
        if format == "ascii":
            symbols = {
                Badge.GOLD: "[G]",
                Badge.SILVER: "[S]",
                Badge.BRONZE: "[B]",
                Badge.CAUTION: "[!]",
                Badge.NOT_RECOMMENDED: "[X]",
                Badge.NONE: "[-]",
            }
        else:
            symbols = {
                Badge.GOLD: "\u2b50",  # Star
                Badge.SILVER: "\u2b24",  # Circle
                Badge.BRONZE: "\u2b25",  # Diamond
                Badge.CAUTION: "\u26a0",  # Warning
                Badge.NOT_RECOMMENDED: "\u274c",  # Cross
                Badge.NONE: "\u2013",  # En dash
            }
        return symbols.get(badge, badge.value)

    def _format_markdown_table(
        self,
        header: list[str],
        rows: list[list[str]],
        widths: list[int],
    ) -> str:
        """Format as Markdown table."""
        lines = []

        # Header
        header_line = (
            "| " + " | ".join(h.ljust(w) for h, w in zip(header, widths, strict=True)) + " |"
        )
        lines.append(header_line)

        # Separator
        sep_line = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
        lines.append(sep_line)

        # Rows
        for row in rows:
            row_line = (
                "| " + " | ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)) + " |"
            )
            lines.append(row_line)

        return "\n".join(lines)

    def _format_ascii_table(
        self,
        header: list[str],
        rows: list[list[str]],
        widths: list[int],
    ) -> str:
        """Format as ASCII table."""
        lines = []

        # Top border
        top_border = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
        lines.append(top_border)

        # Header
        header_line = (
            "| " + " | ".join(h.ljust(w) for h, w in zip(header, widths, strict=True)) + " |"
        )
        lines.append(header_line)

        # Separator
        lines.append(top_border)

        # Rows
        for row in rows:
            row_line = (
                "| " + " | ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)) + " |"
            )
            lines.append(row_line)

        # Bottom border
        lines.append(top_border)

        return "\n".join(lines)

    def _format_unicode_table(
        self,
        header: list[str],
        rows: list[list[str]],
        widths: list[int],
    ) -> str:
        """Format as Unicode box-drawing table."""
        lines = []

        # Top border
        top_border = "\u250c" + "\u252c".join("\u2500" * (w + 2) for w in widths) + "\u2510"
        lines.append(top_border)

        # Header
        header_line = (
            "\u2502 "
            + " \u2502 ".join(h.ljust(w) for h, w in zip(header, widths, strict=True))
            + " \u2502"
        )
        lines.append(header_line)

        # Header separator
        mid_border = "\u251c" + "\u253c".join("\u2500" * (w + 2) for w in widths) + "\u2524"
        lines.append(mid_border)

        # Rows
        for row in rows:
            row_line = (
                "\u2502 "
                + " \u2502 ".join(c.ljust(w) for c, w in zip(row, widths, strict=True))
                + " \u2502"
            )
            lines.append(row_line)

        # Bottom border
        bottom_border = "\u2514" + "\u2534".join("\u2500" * (w + 2) for w in widths) + "\u2518"
        lines.append(bottom_border)

        return "\n".join(lines)


@dataclass
class LeaderboardMetadata:
    """Metadata for the leaderboard."""

    benchmark_version: str
    judge_id: str
    total_models: int
    total_prompts: int
    generated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "benchmark_version": self.benchmark_version,
            "judge_id": self.judge_id,
            "total_models": self.total_models,
            "total_prompts": self.total_prompts,
            "generated_at": self.generated_at.isoformat(),
        }


# ============================================================================
# Convenience Functions
# ============================================================================


def create_leaderboard(
    results: Sequence[EvaluationResultData],
    statistics: dict[str, dict[str, int]] | None = None,
    benchmark_version: str = "base",
    judge_id: str = "",
    total_prompts: int = 0,
) -> LeaderboardGenerator:
    """Create a leaderboard from evaluation results.

    Args:
        results: Sequence of evaluation results.
        statistics: Optional dict mapping model_id to {wins, losses, ties}.
        benchmark_version: Version of benchmark used.
        judge_id: Judge model ID.
        total_prompts: Total prompts evaluated.

    Returns:
        Finalized LeaderboardGenerator.
    """
    generator = LeaderboardGenerator()
    generator.add_from_results(results, statistics)
    generator.finalize(
        benchmark_version=benchmark_version,
        judge_id=judge_id,
        total_prompts=total_prompts,
    )
    return generator
