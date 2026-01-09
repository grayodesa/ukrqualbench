"""Report generation for leaderboard and detailed analysis.

This module provides report generation capabilities:
- Leaderboard generation with badges and rankings
- HTML reports using Jinja2 templates
- Detailed analysis reports in JSON and Markdown
"""

from ukrqualbench.reports.analysis import (
    AnalysisGenerator,
    BenchmarkSummary,
    ComparisonAnalysis,
    ModelAnalysis,
    analyze_results,
    generate_full_report,
)
from ukrqualbench.reports.html import (
    BADGE_CONFIG,
    HTMLReportGenerator,
    generate_calibration_html,
    generate_leaderboard_html,
)
from ukrqualbench.reports.leaderboard import (
    BADGE_THRESHOLDS,
    LeaderboardEntry,
    LeaderboardGenerator,
    LeaderboardMetadata,
    assign_badge,
    create_leaderboard,
)

__all__ = [
    "BADGE_CONFIG",
    "BADGE_THRESHOLDS",
    "AnalysisGenerator",
    "BenchmarkSummary",
    "ComparisonAnalysis",
    "HTMLReportGenerator",
    "LeaderboardEntry",
    "LeaderboardGenerator",
    "LeaderboardMetadata",
    "ModelAnalysis",
    "analyze_results",
    "assign_badge",
    "create_leaderboard",
    "generate_calibration_html",
    "generate_full_report",
    "generate_leaderboard_html",
]
