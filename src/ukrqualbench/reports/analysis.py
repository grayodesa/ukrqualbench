"""Detailed analysis reports for evaluation results.

Generates comprehensive analysis including:
- Model-by-model detailed breakdowns
- Cross-model comparisons
- Trend analysis over evaluation rounds
- Error pattern analysis
- Recommendations based on results
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ukrqualbench.core.schemas import (
    Badge,
    BlockAScores,
    BlockBScores,
    BlockVScores,
    EvaluationResultData,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


# ============================================================================
# Analysis Data Structures
# ============================================================================


@dataclass
class ModelAnalysis:
    """Detailed analysis for a single model."""

    model_id: str
    overall_rating: str  # "excellent", "good", "acceptable", "poor"
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    # Score breakdown
    elo_rating: float = 1500.0
    badge: Badge = Badge.NONE

    # Block A analysis
    block_a_summary: str = ""
    mc_performance: str = ""
    gec_performance: str = ""
    translation_performance: str = ""

    # Block B analysis
    block_b_summary: str = ""
    generation_quality: str = ""
    adversarial_resistance: str = ""
    long_context_stability: str = ""

    # Block V analysis
    block_v_summary: str = ""
    russism_analysis: str = ""
    anglicism_analysis: str = ""
    marker_analysis: str = ""
    fertility_analysis: str = ""

    # Statistics
    win_rate: float = 0.0
    total_comparisons: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "overall_rating": self.overall_rating,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "recommendations": self.recommendations,
            "elo_rating": self.elo_rating,
            "badge": self.badge.value,
            "block_a": {
                "summary": self.block_a_summary,
                "mc_performance": self.mc_performance,
                "gec_performance": self.gec_performance,
                "translation_performance": self.translation_performance,
            },
            "block_b": {
                "summary": self.block_b_summary,
                "generation_quality": self.generation_quality,
                "adversarial_resistance": self.adversarial_resistance,
                "long_context_stability": self.long_context_stability,
            },
            "block_v": {
                "summary": self.block_v_summary,
                "russism_analysis": self.russism_analysis,
                "anglicism_analysis": self.anglicism_analysis,
                "marker_analysis": self.marker_analysis,
                "fertility_analysis": self.fertility_analysis,
            },
            "statistics": {
                "win_rate": self.win_rate,
                "total_comparisons": self.total_comparisons,
            },
        }


@dataclass
class ComparisonAnalysis:
    """Analysis of head-to-head comparison between two models."""

    model_a: str
    model_b: str
    winner: str  # model_a, model_b, or "tie"
    margin: float  # ELO difference
    categories_won_a: list[str] = field(default_factory=list)
    categories_won_b: list[str] = field(default_factory=list)
    key_differences: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_a": self.model_a,
            "model_b": self.model_b,
            "winner": self.winner,
            "margin": self.margin,
            "categories_won_a": self.categories_won_a,
            "categories_won_b": self.categories_won_b,
            "key_differences": self.key_differences,
        }


@dataclass
class BenchmarkSummary:
    """Summary statistics for entire benchmark run."""

    benchmark_version: str
    judge_id: str
    total_models: int
    total_prompts: int
    total_comparisons: int
    runtime_minutes: float
    total_cost_usd: float

    # Distribution stats
    mean_elo: float = 1500.0
    std_elo: float = 0.0
    min_elo: float = 1500.0
    max_elo: float = 1500.0

    # Badge distribution
    gold_count: int = 0
    silver_count: int = 0
    bronze_count: int = 0
    caution_count: int = 0
    not_recommended_count: int = 0

    # Quality metrics averages
    avg_russism_rate: float = 0.0
    avg_positive_markers: float = 0.0
    avg_fertility: float = 0.0

    generated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "benchmark_version": self.benchmark_version,
            "judge_id": self.judge_id,
            "total_models": self.total_models,
            "total_prompts": self.total_prompts,
            "total_comparisons": self.total_comparisons,
            "runtime_minutes": round(self.runtime_minutes, 2),
            "total_cost_usd": round(self.total_cost_usd, 4),
            "elo_distribution": {
                "mean": round(self.mean_elo, 1),
                "std": round(self.std_elo, 1),
                "min": round(self.min_elo, 1),
                "max": round(self.max_elo, 1),
            },
            "badge_distribution": {
                "gold": self.gold_count,
                "silver": self.silver_count,
                "bronze": self.bronze_count,
                "caution": self.caution_count,
                "not_recommended": self.not_recommended_count,
            },
            "quality_averages": {
                "russism_rate": round(self.avg_russism_rate, 2),
                "positive_markers": round(self.avg_positive_markers, 2),
                "fertility": round(self.avg_fertility, 2),
            },
            "generated_at": self.generated_at.isoformat(),
        }


# ============================================================================
# Analysis Generator
# ============================================================================


class AnalysisGenerator:
    """Generator for detailed analysis reports."""

    def __init__(self) -> None:
        """Initialize the analysis generator."""
        self._model_analyses: dict[str, ModelAnalysis] = {}
        self._comparison_analyses: list[ComparisonAnalysis] = []
        self._summary: BenchmarkSummary | None = None

    def analyze_model(
        self,
        result: EvaluationResultData,
        wins: int = 0,
        losses: int = 0,
        ties: int = 0,
    ) -> ModelAnalysis:
        """Generate detailed analysis for a model.

        Args:
            result: Evaluation result for the model.
            wins: Number of pairwise wins.
            losses: Number of pairwise losses.
            ties: Number of ties.

        Returns:
            Detailed model analysis.
        """
        scores = result.scores
        total_games = wins + losses + ties
        win_rate = wins / total_games if total_games > 0 else 0.0

        # Determine overall rating
        overall_rating = self._rate_overall(scores.elo_rating, scores.block_v)

        # Analyze strengths and weaknesses
        strengths, weaknesses = self._analyze_strengths_weaknesses(scores)

        # Generate recommendations
        recommendations = self._generate_recommendations(scores, weaknesses)

        analysis = ModelAnalysis(
            model_id=result.model_id,
            overall_rating=overall_rating,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations,
            elo_rating=scores.elo_rating,
            badge=scores.badge,
            block_a_summary=self._summarize_block_a(scores.block_a),
            mc_performance=self._rate_metric(scores.block_a.mc_accuracy, 0.85, 0.70),
            gec_performance=self._rate_metric(scores.block_a.gec_f1, 0.80, 0.60),
            translation_performance=self._rate_metric(scores.block_a.translation_comet, 0.80, 0.60),
            block_b_summary=self._summarize_block_b(scores.block_b),
            generation_quality=self._rate_elo(scores.block_b.generation_elo),
            adversarial_resistance=self._rate_elo(scores.block_b.adversarial_elo),
            long_context_stability=self._rate_elo(scores.block_b.long_context_elo),
            block_v_summary=self._summarize_block_v(scores.block_v),
            russism_analysis=self._analyze_russisms(scores.block_v.russism_rate),
            anglicism_analysis=self._analyze_anglicisms(scores.block_v.anglicism_rate),
            marker_analysis=self._analyze_markers(scores.block_v.positive_markers),
            fertility_analysis=self._analyze_fertility(scores.block_v.fertility_rate),
            win_rate=win_rate,
            total_comparisons=result.comparisons_count,
        )

        self._model_analyses[result.model_id] = analysis
        return analysis

    def analyze_comparison(
        self,
        result_a: EvaluationResultData,
        result_b: EvaluationResultData,
    ) -> ComparisonAnalysis:
        """Generate head-to-head comparison analysis.

        Args:
            result_a: First model's results.
            result_b: Second model's results.

        Returns:
            Comparison analysis.
        """
        elo_a = result_a.scores.elo_rating
        elo_b = result_b.scores.elo_rating
        margin = abs(elo_a - elo_b)

        if elo_a > elo_b + 20:
            winner = result_a.model_id
        elif elo_b > elo_a + 20:
            winner = result_b.model_id
        else:
            winner = "tie"

        # Analyze category wins
        categories_a, categories_b = self._analyze_category_wins(result_a, result_b)

        # Key differences
        key_differences = self._find_key_differences(result_a, result_b)

        analysis = ComparisonAnalysis(
            model_a=result_a.model_id,
            model_b=result_b.model_id,
            winner=winner,
            margin=margin,
            categories_won_a=categories_a,
            categories_won_b=categories_b,
            key_differences=key_differences,
        )

        self._comparison_analyses.append(analysis)
        return analysis

    def generate_summary(
        self,
        results: Sequence[EvaluationResultData],
        benchmark_version: str = "base",
        judge_id: str = "",
    ) -> BenchmarkSummary:
        """Generate benchmark summary statistics.

        Args:
            results: All evaluation results.
            benchmark_version: Version of benchmark used.
            judge_id: Judge model ID.

        Returns:
            Benchmark summary.
        """
        if not results:
            self._summary = BenchmarkSummary(
                benchmark_version=benchmark_version,
                judge_id=judge_id,
                total_models=0,
                total_prompts=0,
                total_comparisons=0,
                runtime_minutes=0.0,
                total_cost_usd=0.0,
            )
            return self._summary

        # Calculate ELO statistics
        elo_ratings = [r.scores.elo_rating for r in results]
        mean_elo = sum(elo_ratings) / len(elo_ratings)
        std_elo = (sum((e - mean_elo) ** 2 for e in elo_ratings) / len(elo_ratings)) ** 0.5

        # Badge distribution
        badges = [r.scores.badge for r in results]
        gold_count = sum(1 for b in badges if b == Badge.GOLD)
        silver_count = sum(1 for b in badges if b == Badge.SILVER)
        bronze_count = sum(1 for b in badges if b == Badge.BRONZE)
        caution_count = sum(1 for b in badges if b == Badge.CAUTION)
        not_recommended_count = sum(1 for b in badges if b == Badge.NOT_RECOMMENDED)

        # Quality metric averages
        russism_rates = [r.scores.block_v.russism_rate for r in results]
        marker_rates = [r.scores.block_v.positive_markers for r in results]
        fertility_rates = [r.scores.block_v.fertility_rate for r in results]

        # Aggregate metadata
        total_prompts = sum(r.metadata.total_prompts for r in results)
        total_comparisons = sum(r.metadata.total_comparisons for r in results)
        total_runtime = sum(r.metadata.runtime_minutes for r in results)
        total_cost = sum(r.metadata.total_cost_usd for r in results)

        self._summary = BenchmarkSummary(
            benchmark_version=benchmark_version,
            judge_id=judge_id,
            total_models=len(results),
            total_prompts=total_prompts,
            total_comparisons=total_comparisons,
            runtime_minutes=total_runtime,
            total_cost_usd=total_cost,
            mean_elo=mean_elo,
            std_elo=std_elo,
            min_elo=min(elo_ratings),
            max_elo=max(elo_ratings),
            gold_count=gold_count,
            silver_count=silver_count,
            bronze_count=bronze_count,
            caution_count=caution_count,
            not_recommended_count=not_recommended_count,
            avg_russism_rate=sum(russism_rates) / len(russism_rates),
            avg_positive_markers=sum(marker_rates) / len(marker_rates),
            avg_fertility=sum(fertility_rates) / len(fertility_rates),
        )

        return self._summary

    def to_json(self, indent: int = 2) -> str:
        """Export analysis to JSON.

        Args:
            indent: JSON indentation.

        Returns:
            JSON string.
        """
        data: dict[str, Any] = {
            "models": {
                model_id: analysis.to_dict()
                for model_id, analysis in self._model_analyses.items()
            },
            "comparisons": [c.to_dict() for c in self._comparison_analyses],
        }

        if self._summary:
            data["summary"] = self._summary.to_dict()

        return json.dumps(data, indent=indent, default=str, ensure_ascii=False)

    def save_report(
        self,
        output_path: str | Path,
        format: str = "json",
    ) -> Path:
        """Save analysis report to file.

        Args:
            output_path: Output file path.
            format: Output format ("json" or "md").

        Returns:
            Path to saved file.
        """
        output_path = Path(output_path)

        if format == "json":
            content = self.to_json()
        elif format == "md":
            content = self._to_markdown()
        else:
            raise ValueError(f"Unsupported format: {format}")

        output_path.write_text(content, encoding="utf-8")
        return output_path

    # ========================================================================
    # Private Analysis Methods
    # ========================================================================

    def _rate_overall(self, elo: float, block_v: BlockVScores) -> str:
        """Rate overall model quality."""
        if elo >= 1650 and block_v.russism_rate < 1.0:
            return "excellent"
        elif elo >= 1550 and block_v.russism_rate < 3.0:
            return "good"
        elif elo >= 1450:
            return "acceptable"
        else:
            return "poor"

    def _rate_metric(self, value: float, good_threshold: float, acceptable_threshold: float) -> str:
        """Rate a metric value."""
        if value >= good_threshold:
            return "excellent"
        elif value >= acceptable_threshold:
            return "good"
        elif value >= acceptable_threshold * 0.8:
            return "acceptable"
        else:
            return "poor"

    def _rate_elo(self, elo: float) -> str:
        """Rate an ELO score."""
        if elo >= 1600:
            return "excellent"
        elif elo >= 1550:
            return "good"
        elif elo >= 1450:
            return "acceptable"
        else:
            return "poor"

    def _analyze_strengths_weaknesses(
        self, scores: Any
    ) -> tuple[list[str], list[str]]:
        """Identify model strengths and weaknesses."""
        strengths = []
        weaknesses = []

        # Block A
        if scores.block_a.mc_accuracy >= 0.90:
            strengths.append("Excellent multiple choice accuracy")
        elif scores.block_a.mc_accuracy < 0.70:
            weaknesses.append("Low multiple choice accuracy")

        if scores.block_a.gec_f1 >= 0.85:
            strengths.append("Strong grammar error correction")
        elif scores.block_a.gec_f1 < 0.60:
            weaknesses.append("Weak grammar error correction")

        # Block V
        if scores.block_v.russism_rate < 1.0:
            strengths.append("Very low russism rate")
        elif scores.block_v.russism_rate > 5.0:
            weaknesses.append("High russism rate")

        if scores.block_v.positive_markers > 5.0:
            strengths.append("Rich use of native Ukrainian markers")
        elif scores.block_v.positive_markers < 1.0:
            weaknesses.append("Lacks native Ukrainian markers")

        if scores.block_v.fertility_rate < 1.5:
            strengths.append("Efficient tokenization")
        elif scores.block_v.fertility_rate > 2.0:
            weaknesses.append("Poor tokenization efficiency")

        return strengths, weaknesses

    def _generate_recommendations(
        self, scores: Any, weaknesses: list[str]
    ) -> list[str]:
        """Generate recommendations based on weaknesses."""
        recommendations = []

        if scores.block_v.russism_rate > 3.0:
            recommendations.append(
                "Consider fine-tuning on clean Ukrainian corpora to reduce russisms"
            )

        if scores.block_v.positive_markers < 2.0:
            recommendations.append(
                "Train on authentic Ukrainian texts to improve native marker usage"
            )

        if scores.block_v.fertility_rate > 2.0:
            recommendations.append(
                "Review tokenizer for Ukrainian language optimization"
            )

        if scores.block_a.gec_f1 < 0.70:
            recommendations.append(
                "Additional training on grammar error correction data may help"
            )

        return recommendations

    def _summarize_block_a(self, block_a: BlockAScores) -> str:
        """Summarize Block A performance."""
        avg = (block_a.mc_accuracy + block_a.gec_f1 + block_a.translation_comet) / 3
        if avg >= 0.85:
            return "Excellent calibration test performance"
        elif avg >= 0.70:
            return "Good calibration test performance"
        elif avg >= 0.55:
            return "Acceptable calibration test performance"
        else:
            return "Below average calibration test performance"

    def _summarize_block_b(self, block_b: BlockBScores) -> str:
        """Summarize Block B performance."""
        avg_elo = (
            block_b.generation_elo + block_b.adversarial_elo + block_b.long_context_elo
        ) / 3
        if avg_elo >= 1600:
            return "Excellent generation quality across all tasks"
        elif avg_elo >= 1550:
            return "Good generation quality"
        elif avg_elo >= 1450:
            return "Acceptable generation quality"
        else:
            return "Below average generation quality"

    def _summarize_block_v(self, block_v: BlockVScores) -> str:
        """Summarize Block V metrics."""
        if block_v.russism_rate < 2.0 and block_v.positive_markers > 3.0:
            return "High-quality native Ukrainian output"
        elif block_v.russism_rate < 5.0:
            return "Generally acceptable Ukrainian quality"
        else:
            return "Ukrainian quality needs improvement"

    def _analyze_russisms(self, rate: float) -> str:
        """Analyze russism rate."""
        if rate < 1.0:
            return f"Excellent: {rate:.2f} russisms per 1K tokens (well below threshold)"
        elif rate < 3.0:
            return f"Good: {rate:.2f} russisms per 1K tokens (below threshold)"
        elif rate < 5.0:
            return f"Acceptable: {rate:.2f} russisms per 1K tokens (approaching threshold)"
        else:
            return f"Poor: {rate:.2f} russisms per 1K tokens (above acceptable threshold)"

    def _analyze_anglicisms(self, rate: float) -> str:
        """Analyze anglicism rate."""
        if rate < 2.0:
            return f"Excellent: {rate:.2f} anglicisms per 1K tokens"
        elif rate < 5.0:
            return f"Good: {rate:.2f} anglicisms per 1K tokens"
        else:
            return f"High: {rate:.2f} anglicisms per 1K tokens"

    def _analyze_markers(self, rate: float) -> str:
        """Analyze positive marker rate."""
        if rate > 5.0:
            return f"Excellent: {rate:.2f} native markers per 1K tokens (rich usage)"
        elif rate > 3.0:
            return f"Good: {rate:.2f} native markers per 1K tokens"
        elif rate > 1.0:
            return f"Acceptable: {rate:.2f} native markers per 1K tokens"
        else:
            return f"Low: {rate:.2f} native markers per 1K tokens (needs improvement)"

    def _analyze_fertility(self, rate: float) -> str:
        """Analyze fertility rate."""
        if rate < 1.3:
            return f"Excellent: {rate:.2f} tokens per word (efficient tokenization)"
        elif rate < 1.5:
            return f"Good: {rate:.2f} tokens per word"
        elif rate < 2.0:
            return f"Acceptable: {rate:.2f} tokens per word"
        else:
            return f"Poor: {rate:.2f} tokens per word (inefficient tokenization)"

    def _analyze_category_wins(
        self,
        result_a: EvaluationResultData,
        result_b: EvaluationResultData,
    ) -> tuple[list[str], list[str]]:
        """Analyze which categories each model won."""
        categories_a = []
        categories_b = []

        # Block A comparisons
        if result_a.scores.block_a.mc_accuracy > result_b.scores.block_a.mc_accuracy + 0.05:
            categories_a.append("Multiple Choice")
        elif result_b.scores.block_a.mc_accuracy > result_a.scores.block_a.mc_accuracy + 0.05:
            categories_b.append("Multiple Choice")

        if result_a.scores.block_a.gec_f1 > result_b.scores.block_a.gec_f1 + 0.05:
            categories_a.append("Grammar Error Correction")
        elif result_b.scores.block_a.gec_f1 > result_a.scores.block_a.gec_f1 + 0.05:
            categories_b.append("Grammar Error Correction")

        # Block V comparisons
        if result_a.scores.block_v.russism_rate < result_b.scores.block_v.russism_rate - 0.5:
            categories_a.append("Russism Avoidance")
        elif result_b.scores.block_v.russism_rate < result_a.scores.block_v.russism_rate - 0.5:
            categories_b.append("Russism Avoidance")

        if result_a.scores.block_v.positive_markers > result_b.scores.block_v.positive_markers + 0.5:
            categories_a.append("Native Markers")
        elif result_b.scores.block_v.positive_markers > result_a.scores.block_v.positive_markers + 0.5:
            categories_b.append("Native Markers")

        return categories_a, categories_b

    def _find_key_differences(
        self,
        result_a: EvaluationResultData,
        result_b: EvaluationResultData,
    ) -> list[str]:
        """Find key differences between two models."""
        differences = []

        # ELO difference
        elo_diff = result_a.scores.elo_rating - result_b.scores.elo_rating
        if abs(elo_diff) > 50:
            better = result_a.model_id if elo_diff > 0 else result_b.model_id
            differences.append(f"{better} has significantly higher ELO ({abs(elo_diff):.0f} points)")

        # Russism difference
        russ_a = result_a.scores.block_v.russism_rate
        russ_b = result_b.scores.block_v.russism_rate
        if abs(russ_a - russ_b) > 2.0:
            better = result_a.model_id if russ_a < russ_b else result_b.model_id
            differences.append(f"{better} has lower russism rate")

        # Marker difference
        mark_a = result_a.scores.block_v.positive_markers
        mark_b = result_b.scores.block_v.positive_markers
        if abs(mark_a - mark_b) > 2.0:
            better = result_a.model_id if mark_a > mark_b else result_b.model_id
            differences.append(f"{better} uses more native markers")

        return differences

    def _to_markdown(self) -> str:
        """Export analysis to Markdown format."""
        lines = ["# UkrQualBench Analysis Report\n"]

        if self._summary:
            lines.append("## Summary\n")
            lines.append(f"- **Benchmark Version:** {self._summary.benchmark_version}")
            lines.append(f"- **Judge:** {self._summary.judge_id}")
            lines.append(f"- **Models Evaluated:** {self._summary.total_models}")
            lines.append(f"- **Total Comparisons:** {self._summary.total_comparisons}")
            lines.append(f"- **Runtime:** {self._summary.runtime_minutes:.1f} minutes")
            lines.append(f"- **Total Cost:** ${self._summary.total_cost_usd:.2f}")
            lines.append("")

        lines.append("## Model Analyses\n")
        for model_id, analysis in self._model_analyses.items():
            lines.append(f"### {model_id}\n")
            lines.append(f"**Overall Rating:** {analysis.overall_rating.upper()}")
            lines.append(f"**ELO:** {analysis.elo_rating:.1f} ({analysis.badge.value})")
            lines.append("")

            if analysis.strengths:
                lines.append("**Strengths:**")
                for s in analysis.strengths:
                    lines.append(f"- {s}")
                lines.append("")

            if analysis.weaknesses:
                lines.append("**Weaknesses:**")
                for w in analysis.weaknesses:
                    lines.append(f"- {w}")
                lines.append("")

            if analysis.recommendations:
                lines.append("**Recommendations:**")
                for r in analysis.recommendations:
                    lines.append(f"- {r}")
                lines.append("")

        return "\n".join(lines)


# ============================================================================
# Convenience Functions
# ============================================================================


def analyze_results(
    results: Sequence[EvaluationResultData],
    statistics: dict[str, dict[str, int]] | None = None,
    benchmark_version: str = "base",
    judge_id: str = "",
) -> AnalysisGenerator:
    """Analyze evaluation results.

    Args:
        results: Sequence of evaluation results.
        statistics: Optional dict mapping model_id to {wins, losses, ties}.
        benchmark_version: Version of benchmark used.
        judge_id: Judge model ID.

    Returns:
        Analysis generator with results.
    """
    generator = AnalysisGenerator()
    statistics = statistics or {}

    for result in results:
        stats = statistics.get(result.model_id, {})
        generator.analyze_model(
            result,
            wins=stats.get("wins", 0),
            losses=stats.get("losses", 0),
            ties=stats.get("ties", 0),
        )

    generator.generate_summary(results, benchmark_version, judge_id)
    return generator


def generate_full_report(
    results: Sequence[EvaluationResultData],
    output_dir: str | Path,
    statistics: dict[str, dict[str, int]] | None = None,
    benchmark_version: str = "base",
    judge_id: str = "",
) -> dict[str, Path]:
    """Generate complete analysis report in multiple formats.

    Args:
        results: Sequence of evaluation results.
        output_dir: Output directory.
        statistics: Optional win/loss/tie statistics.
        benchmark_version: Version of benchmark used.
        judge_id: Judge model ID.

    Returns:
        Dict mapping format to output path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = analyze_results(results, statistics, benchmark_version, judge_id)

    outputs = {
        "json": generator.save_report(output_dir / "analysis.json", "json"),
        "md": generator.save_report(output_dir / "analysis.md", "md"),
    }

    return outputs
