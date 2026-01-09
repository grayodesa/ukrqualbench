"""HTML report generation using Jinja2 templates.

Generates HTML reports including:
- Interactive leaderboard tables with sorting
- Model comparison charts
- Block-by-block score breakdowns
- ELO rating progression visualization
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from jinja2 import Environment, PackageLoader, select_autoescape

from ukrqualbench.core.schemas import Badge, CalibrationResultData, EvaluationResultData

if TYPE_CHECKING:
    from ukrqualbench.reports.leaderboard import LeaderboardGenerator


# ============================================================================
# Badge Configuration
# ============================================================================

BADGE_CONFIG = {
    Badge.GOLD: {
        "color": "#FFD700",
        "bg_color": "#FFF9E6",
        "icon": "\u2b50",
        "label": "Gold",
        "description": "Excellent Ukrainian language quality",
    },
    Badge.SILVER: {
        "color": "#C0C0C0",
        "bg_color": "#F5F5F5",
        "icon": "\u26aa",
        "label": "Silver",
        "description": "Very good Ukrainian language quality",
    },
    Badge.BRONZE: {
        "color": "#CD7F32",
        "bg_color": "#FDF5E6",
        "icon": "\u26ab",
        "label": "Bronze",
        "description": "Good Ukrainian language quality",
    },
    Badge.CAUTION: {
        "color": "#FFA500",
        "bg_color": "#FFF3E0",
        "icon": "\u26a0\ufe0f",
        "label": "Caution",
        "description": "Some language quality concerns",
    },
    Badge.NOT_RECOMMENDED: {
        "color": "#FF4444",
        "bg_color": "#FFEBEE",
        "icon": "\u274c",
        "label": "Not Recommended",
        "description": "Significant language quality issues",
    },
    Badge.NONE: {
        "color": "#888888",
        "bg_color": "#F0F0F0",
        "icon": "\u2796",
        "label": "Unrated",
        "description": "Not enough data for rating",
    },
}


# ============================================================================
# HTML Templates (inline for simplicity, can be moved to files)
# ============================================================================

LEADERBOARD_TEMPLATE = """<!DOCTYPE html>
<html lang="uk">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>UkrQualBench Leaderboard</title>
    <style>
        :root {
            --primary: #2563eb;
            --secondary: #64748b;
            --success: #22c55e;
            --warning: #f59e0b;
            --danger: #ef4444;
            --bg: #ffffff;
            --bg-secondary: #f8fafc;
            --text: #1e293b;
            --border: #e2e8f0;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-secondary);
            color: var(--text);
            line-height: 1.6;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }

        header {
            text-align: center;
            margin-bottom: 2rem;
        }

        h1 {
            font-size: 2.5rem;
            color: var(--text);
            margin-bottom: 0.5rem;
        }

        .subtitle {
            color: var(--secondary);
            font-size: 1.1rem;
        }

        .metadata {
            display: flex;
            justify-content: center;
            gap: 2rem;
            margin-top: 1rem;
            flex-wrap: wrap;
        }

        .metadata-item {
            background: var(--bg);
            padding: 0.5rem 1rem;
            border-radius: 8px;
            border: 1px solid var(--border);
        }

        .metadata-label {
            font-size: 0.8rem;
            color: var(--secondary);
            text-transform: uppercase;
        }

        .metadata-value {
            font-weight: 600;
            color: var(--text);
        }

        .leaderboard-table {
            width: 100%;
            background: var(--bg);
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            overflow: hidden;
            margin-top: 2rem;
        }

        table {
            width: 100%;
            border-collapse: collapse;
        }

        th, td {
            padding: 1rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }

        th {
            background: var(--bg-secondary);
            font-weight: 600;
            color: var(--secondary);
            text-transform: uppercase;
            font-size: 0.8rem;
            letter-spacing: 0.05em;
            cursor: pointer;
            user-select: none;
        }

        th:hover {
            background: #e2e8f0;
        }

        th.sorted-asc::after {
            content: ' \\25B2';
        }

        th.sorted-desc::after {
            content: ' \\25BC';
        }

        tr:hover {
            background: var(--bg-secondary);
        }

        .rank {
            font-weight: 700;
            color: var(--primary);
            width: 60px;
        }

        .rank-1 { color: #FFD700; }
        .rank-2 { color: #C0C0C0; }
        .rank-3 { color: #CD7F32; }

        .model-name {
            font-weight: 600;
        }

        .elo {
            font-family: 'Monaco', 'Consolas', monospace;
            font-weight: 700;
            font-size: 1.1rem;
        }

        .badge {
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 500;
        }

        .metric {
            font-family: 'Monaco', 'Consolas', monospace;
            font-size: 0.9rem;
        }

        .metric-good { color: var(--success); }
        .metric-warning { color: var(--warning); }
        .metric-bad { color: var(--danger); }

        .block-scores {
            display: none;
        }

        .expand-btn {
            background: none;
            border: 1px solid var(--border);
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.8rem;
        }

        .expand-btn:hover {
            background: var(--bg-secondary);
        }

        .progress-bar {
            width: 100px;
            height: 8px;
            background: var(--border);
            border-radius: 4px;
            overflow: hidden;
            display: inline-block;
            vertical-align: middle;
            margin-left: 0.5rem;
        }

        .progress-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s ease;
        }

        footer {
            text-align: center;
            margin-top: 2rem;
            padding: 1rem;
            color: var(--secondary);
            font-size: 0.9rem;
        }

        footer a {
            color: var(--primary);
            text-decoration: none;
        }

        @media (max-width: 768px) {
            .container {
                padding: 1rem;
            }

            h1 {
                font-size: 1.5rem;
            }

            th, td {
                padding: 0.5rem;
                font-size: 0.85rem;
            }

            .metadata {
                flex-direction: column;
                gap: 0.5rem;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>UkrQualBench Leaderboard</h1>
            <p class="subtitle">Ukrainian Language Quality Benchmark for LLMs</p>
            <div class="metadata">
                <div class="metadata-item">
                    <div class="metadata-label">Benchmark</div>
                    <div class="metadata-value">{{ metadata.benchmark_version }}</div>
                </div>
                <div class="metadata-item">
                    <div class="metadata-label">Judge</div>
                    <div class="metadata-value">{{ metadata.judge_id or 'N/A' }}</div>
                </div>
                <div class="metadata-item">
                    <div class="metadata-label">Models</div>
                    <div class="metadata-value">{{ metadata.total_models }}</div>
                </div>
                <div class="metadata-item">
                    <div class="metadata-label">Generated</div>
                    <div class="metadata-value">{{ metadata.generated_at }}</div>
                </div>
            </div>
        </header>

        <div class="leaderboard-table">
            <table id="leaderboard">
                <thead>
                    <tr>
                        <th data-sort="rank" class="sorted-asc">Rank</th>
                        <th data-sort="model">Model</th>
                        <th data-sort="elo">ELO</th>
                        <th data-sort="badge">Badge</th>
                        <th data-sort="russism">Russisms</th>
                        <th data-sort="markers">Markers</th>
                        <th data-sort="fertility">Fertility</th>
                        <th>Details</th>
                    </tr>
                </thead>
                <tbody>
                    {% for entry in entries %}
                    <tr data-model="{{ entry.model_id }}">
                        <td class="rank rank-{{ entry.rank }}">{{ entry.rank }}</td>
                        <td class="model-name">{{ entry.model_id }}</td>
                        <td class="elo">{{ entry.elo_rating|round(1) }}</td>
                        <td>
                            <span class="badge" style="background: {{ badge_config[entry.badge].bg_color }}; color: {{ badge_config[entry.badge].color }};">
                                {{ badge_config[entry.badge].icon }} {{ badge_config[entry.badge].label }}
                            </span>
                        </td>
                        <td class="metric {{ 'metric-good' if entry.russism_rate < 3 else 'metric-warning' if entry.russism_rate < 5 else 'metric-bad' }}">
                            {{ entry.russism_rate|round(2) if entry.russism_rate is not none else 'N/A' }}
                            {% if entry.russism_rate is not none %}
                            <div class="progress-bar">
                                <div class="progress-fill" style="width: {{ [entry.russism_rate * 10, 100]|min }}%; background: {{ '#22c55e' if entry.russism_rate < 3 else '#f59e0b' if entry.russism_rate < 5 else '#ef4444' }};"></div>
                            </div>
                            {% endif %}
                        </td>
                        <td class="metric {{ 'metric-good' if entry.positive_markers and entry.positive_markers > 3 else 'metric-warning' if entry.positive_markers and entry.positive_markers > 1 else 'metric-bad' }}">
                            {{ entry.positive_markers|round(2) if entry.positive_markers is not none else 'N/A' }}
                        </td>
                        <td class="metric {{ 'metric-good' if entry.fertility_rate and entry.fertility_rate < 1.5 else 'metric-warning' if entry.fertility_rate and entry.fertility_rate < 2.0 else 'metric-bad' }}">
                            {{ entry.fertility_rate|round(2) if entry.fertility_rate is not none else 'N/A' }}
                        </td>
                        <td>
                            <button class="expand-btn" onclick="toggleDetails('{{ entry.model_id }}')">View</button>
                        </td>
                    </tr>
                    <tr class="block-scores" id="details-{{ entry.model_id|replace('.', '-')|replace('/', '-') }}">
                        <td colspan="8">
                            <div style="padding: 1rem; background: #f8fafc;">
                                <strong>Block A (Calibration):</strong>
                                MC Accuracy: {{ (entry.mc_accuracy * 100)|round(1) if entry.mc_accuracy is not none else 'N/A' }}% |
                                GEC F1: {{ (entry.gec_f1 * 100)|round(1) if entry.gec_f1 is not none else 'N/A' }}% |
                                Translation COMET: {{ (entry.translation_comet * 100)|round(1) if entry.translation_comet is not none else 'N/A' }}%
                                <br>
                                <strong>Block B (Generation):</strong>
                                Free Gen ELO: {{ entry.generation_elo|round(0) if entry.generation_elo is not none else 'N/A' }} |
                                Adversarial ELO: {{ entry.adversarial_elo|round(0) if entry.adversarial_elo is not none else 'N/A' }} |
                                Long Context ELO: {{ entry.long_context_elo|round(0) if entry.long_context_elo is not none else 'N/A' }}
                                <br>
                                <strong>Statistics:</strong>
                                Wins: {{ entry.wins }} | Losses: {{ entry.losses }} | Ties: {{ entry.ties }}
                            </div>
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>

        <footer>
            <p>Generated by <a href="https://github.com/ukrqualbench/ukrqualbench">UkrQualBench</a></p>
        </footer>
    </div>

    <script>
        function toggleDetails(modelId) {
            const id = 'details-' + modelId.replace(/\\./g, '-').replace(/\\//g, '-');
            const row = document.getElementById(id);
            if (row) {
                row.style.display = row.style.display === 'table-row' ? 'none' : 'table-row';
            }
        }

        // Table sorting
        document.querySelectorAll('th[data-sort]').forEach(th => {
            th.addEventListener('click', () => {
                const table = th.closest('table');
                const tbody = table.querySelector('tbody');
                const rows = Array.from(tbody.querySelectorAll('tr:not(.block-scores)'));
                const sortKey = th.dataset.sort;
                const isAsc = th.classList.contains('sorted-asc');

                // Remove sort classes from all headers
                table.querySelectorAll('th').forEach(h => {
                    h.classList.remove('sorted-asc', 'sorted-desc');
                });

                // Add sort class to clicked header
                th.classList.add(isAsc ? 'sorted-desc' : 'sorted-asc');

                // Sort rows
                rows.sort((a, b) => {
                    let aVal, bVal;

                    switch(sortKey) {
                        case 'rank':
                            aVal = parseInt(a.querySelector('.rank').textContent);
                            bVal = parseInt(b.querySelector('.rank').textContent);
                            break;
                        case 'model':
                            aVal = a.querySelector('.model-name').textContent;
                            bVal = b.querySelector('.model-name').textContent;
                            break;
                        case 'elo':
                            aVal = parseFloat(a.querySelector('.elo').textContent);
                            bVal = parseFloat(b.querySelector('.elo').textContent);
                            break;
                        default:
                            aVal = a.cells[th.cellIndex].textContent;
                            bVal = b.cells[th.cellIndex].textContent;
                    }

                    if (typeof aVal === 'string') {
                        return isAsc ? bVal.localeCompare(aVal) : aVal.localeCompare(bVal);
                    }
                    return isAsc ? bVal - aVal : aVal - bVal;
                });

                // Re-append sorted rows
                rows.forEach(row => {
                    const modelId = row.dataset.model;
                    const detailsId = 'details-' + modelId.replace(/\\./g, '-').replace(/\\//g, '-');
                    const detailsRow = document.getElementById(detailsId);
                    tbody.appendChild(row);
                    if (detailsRow) {
                        tbody.appendChild(detailsRow);
                    }
                });
            });
        });
    </script>
</body>
</html>
"""

CALIBRATION_TEMPLATE = """<!DOCTYPE html>
<html lang="uk">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>UkrQualBench Calibration Report</title>
    <style>
        :root {
            --primary: #2563eb;
            --success: #22c55e;
            --warning: #f59e0b;
            --danger: #ef4444;
            --bg: #ffffff;
            --bg-secondary: #f8fafc;
            --text: #1e293b;
            --border: #e2e8f0;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-secondary);
            color: var(--text);
            line-height: 1.6;
            padding: 2rem;
        }

        .container {
            max-width: 900px;
            margin: 0 auto;
        }

        h1 {
            text-align: center;
            margin-bottom: 2rem;
        }

        .status {
            text-align: center;
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
        }

        .status-pass {
            background: #dcfce7;
            border: 2px solid var(--success);
        }

        .status-fail {
            background: #fee2e2;
            border: 2px solid var(--danger);
        }

        .status-icon {
            font-size: 3rem;
            margin-bottom: 1rem;
        }

        .status-text {
            font-size: 1.5rem;
            font-weight: 700;
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }

        .metric-card {
            background: var(--bg);
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }

        .metric-name {
            font-size: 0.9rem;
            color: var(--text);
            opacity: 0.7;
            margin-bottom: 0.5rem;
        }

        .metric-value {
            font-size: 2rem;
            font-weight: 700;
        }

        .metric-bar {
            height: 8px;
            background: var(--border);
            border-radius: 4px;
            margin-top: 0.5rem;
            overflow: hidden;
        }

        .metric-fill {
            height: 100%;
            border-radius: 4px;
        }

        .threshold {
            font-size: 0.8rem;
            color: var(--text);
            opacity: 0.6;
            margin-top: 0.25rem;
        }

        .failure-reasons {
            background: #fee2e2;
            padding: 1rem;
            border-radius: 8px;
            margin-top: 1rem;
        }

        .failure-reasons h3 {
            color: var(--danger);
            margin-bottom: 0.5rem;
        }

        .failure-reasons ul {
            margin-left: 1.5rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Judge Calibration Report</h1>
        <h2 style="text-align: center; color: var(--primary); margin-bottom: 2rem;">{{ result.judge_id }}</h2>

        <div class="status {{ 'status-pass' if result.passed else 'status-fail' }}">
            <div class="status-icon">{{ '\\u2705' if result.passed else '\\u274c' }}</div>
            <div class="status-text">{{ 'CALIBRATION PASSED' if result.passed else 'CALIBRATION FAILED' }}</div>
            <div>Final Score: {{ (result.final_score * 100)|round(1) }}%</div>
        </div>

        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-name">MC Accuracy</div>
                <div class="metric-value" style="color: {{ '#22c55e' if result.mc_accuracy >= 0.85 else '#ef4444' }}">
                    {{ (result.mc_accuracy * 100)|round(1) }}%
                </div>
                <div class="metric-bar">
                    <div class="metric-fill" style="width: {{ result.mc_accuracy * 100 }}%; background: {{ '#22c55e' if result.mc_accuracy >= 0.85 else '#ef4444' }};"></div>
                </div>
                <div class="threshold">Threshold: 85%</div>
            </div>

            <div class="metric-card">
                <div class="metric-name">GEC F1</div>
                <div class="metric-value" style="color: {{ '#22c55e' if result.gec_f1 >= 0.80 else '#ef4444' }}">
                    {{ (result.gec_f1 * 100)|round(1) }}%
                </div>
                <div class="metric-bar">
                    <div class="metric-fill" style="width: {{ result.gec_f1 * 100 }}%; background: {{ '#22c55e' if result.gec_f1 >= 0.80 else '#ef4444' }};"></div>
                </div>
                <div class="threshold">Threshold: 80%</div>
            </div>

            <div class="metric-card">
                <div class="metric-name">Russism Detection F1</div>
                <div class="metric-value" style="color: {{ '#22c55e' if result.russism_f1 >= 0.85 else '#ef4444' }}">
                    {{ (result.russism_f1 * 100)|round(1) }}%
                </div>
                <div class="metric-bar">
                    <div class="metric-fill" style="width: {{ result.russism_f1 * 100 }}%; background: {{ '#22c55e' if result.russism_f1 >= 0.85 else '#ef4444' }};"></div>
                </div>
                <div class="threshold">Threshold: 85%</div>
            </div>

            <div class="metric-card">
                <div class="metric-name">False Positive Rate</div>
                <div class="metric-value" style="color: {{ '#22c55e' if result.false_positive_rate <= 0.15 else '#ef4444' }}">
                    {{ (result.false_positive_rate * 100)|round(1) }}%
                </div>
                <div class="metric-bar">
                    <div class="metric-fill" style="width: {{ result.false_positive_rate * 100 }}%; background: {{ '#22c55e' if result.false_positive_rate <= 0.15 else '#ef4444' }};"></div>
                </div>
                <div class="threshold">Threshold: &le;15%</div>
            </div>

            <div class="metric-card">
                <div class="metric-name">Pairwise Consistency</div>
                <div class="metric-value" style="color: {{ '#22c55e' if result.pairwise_consistency >= 0.90 else '#ef4444' }}">
                    {{ (result.pairwise_consistency * 100)|round(1) }}%
                </div>
                <div class="metric-bar">
                    <div class="metric-fill" style="width: {{ result.pairwise_consistency * 100 }}%; background: {{ '#22c55e' if result.pairwise_consistency >= 0.90 else '#ef4444' }};"></div>
                </div>
                <div class="threshold">Threshold: 90%</div>
            </div>

            <div class="metric-card">
                <div class="metric-name">Position Bias</div>
                <div class="metric-value" style="color: {{ '#22c55e' if result.position_bias <= 0.1 else '#ef4444' }}">
                    {{ (result.position_bias * 100)|round(1) }}%
                </div>
                <div class="metric-bar">
                    <div class="metric-fill" style="width: {{ result.position_bias * 100 }}%; background: {{ '#22c55e' if result.position_bias <= 0.1 else '#ef4444' }};"></div>
                </div>
                <div class="threshold">Threshold: &le;10%</div>
            </div>
        </div>

        {% if result.failure_reasons %}
        <div class="failure-reasons">
            <h3>Failure Reasons</h3>
            <ul>
                {% for reason in result.failure_reasons %}
                <li>{{ reason }}</li>
                {% endfor %}
            </ul>
        </div>
        {% endif %}

        <footer style="text-align: center; margin-top: 2rem; color: var(--text); opacity: 0.6;">
            Generated: {{ result.timestamp.isoformat() if result.timestamp else 'N/A' }}
        </footer>
    </div>
</body>
</html>
"""


# ============================================================================
# HTML Report Generator
# ============================================================================


class HTMLReportGenerator:
    """Generator for HTML reports.

    Uses Jinja2 templates for flexible, customizable output.
    """

    def __init__(self) -> None:
        """Initialize the HTML report generator."""
        self._env = Environment(
            loader=PackageLoader("ukrqualbench.reports", "templates"),
            autoescape=select_autoescape(["html", "xml"]),
        )
        # Add custom filters
        self._env.filters["badge_config"] = lambda badge: BADGE_CONFIG.get(
            badge, BADGE_CONFIG[Badge.NONE]
        )

    def render_leaderboard(
        self,
        leaderboard: LeaderboardGenerator,
        title: str = "UkrQualBench Leaderboard",
    ) -> str:
        """Render leaderboard as HTML.

        Args:
            leaderboard: Leaderboard generator with entries.
            title: Page title.

        Returns:
            HTML string.
        """
        # Use inline template for simplicity
        from jinja2 import Template

        template = Template(LEADERBOARD_TEMPLATE)

        metadata_dict = {
            "benchmark_version": "base",
            "judge_id": "",
            "total_models": len(leaderboard.entries),
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }

        if leaderboard.metadata:
            metadata_dict.update(
                {
                    "benchmark_version": leaderboard.metadata.benchmark_version,
                    "judge_id": leaderboard.metadata.judge_id,
                    "total_models": leaderboard.metadata.total_models,
                    "generated_at": leaderboard.metadata.generated_at.strftime("%Y-%m-%d %H:%M"),
                }
            )

        return template.render(
            title=title,
            entries=leaderboard.entries,
            metadata=metadata_dict,
            badge_config=BADGE_CONFIG,
        )

    def render_calibration_report(
        self,
        result: CalibrationResultData,
    ) -> str:
        """Render calibration report as HTML.

        Args:
            result: Calibration result data.

        Returns:
            HTML string.
        """
        from jinja2 import Template

        template = Template(CALIBRATION_TEMPLATE)

        return template.render(result=result)

    def render_model_comparison(
        self,
        results: list[EvaluationResultData],
        title: str = "Model Comparison",
    ) -> str:
        """Render model comparison as HTML.

        Args:
            results: List of evaluation results to compare.
            title: Page title.

        Returns:
            HTML string.
        """
        # Build comparison data
        models = [r.model_id for r in results]
        elo_scores = [r.scores.elo_rating for r in results]

        html = f"""<!DOCTYPE html>
<html lang="uk">
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: sans-serif; padding: 2rem; background: #f8fafc; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ text-align: center; }}
        .charts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 2rem; }}
        .chart-container {{ background: white; padding: 1rem; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <div class="charts">
            <div class="chart-container">
                <canvas id="eloChart"></canvas>
            </div>
            <div class="chart-container">
                <canvas id="metricsChart"></canvas>
            </div>
        </div>
    </div>
    <script>
        // ELO Chart
        new Chart(document.getElementById('eloChart'), {{
            type: 'bar',
            data: {{
                labels: {json.dumps(models)},
                datasets: [{{
                    label: 'ELO Rating',
                    data: {json.dumps(elo_scores)},
                    backgroundColor: 'rgba(37, 99, 235, 0.8)',
                    borderColor: 'rgba(37, 99, 235, 1)',
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    title: {{ display: true, text: 'ELO Ratings' }}
                }},
                scales: {{
                    y: {{ beginAtZero: false, min: 1300, max: 1700 }}
                }}
            }}
        }});

        // Metrics Chart
        new Chart(document.getElementById('metricsChart'), {{
            type: 'radar',
            data: {{
                labels: ['Low Russisms', 'High Markers', 'Low Fertility'],
                datasets: {
            json.dumps(
                [
                    {
                        "label": model,
                        "data": [
                            10 - min(result.scores.block_v.russism_rate, 10),
                            result.scores.block_v.positive_markers,
                            3 - min(result.scores.block_v.fertility_rate, 3),
                        ],
                    }
                    for model, result in zip(models, results, strict=True)
                ]
            )
        }
            }},
            options: {{
                responsive: true,
                plugins: {{
                    title: {{ display: true, text: 'Quality Metrics' }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
        return html

    def save_leaderboard(
        self,
        leaderboard: LeaderboardGenerator,
        output_path: str | Path,
        title: str = "UkrQualBench Leaderboard",
    ) -> Path:
        """Save leaderboard HTML to file.

        Args:
            leaderboard: Leaderboard generator.
            output_path: Output file path.
            title: Page title.

        Returns:
            Path to saved file.
        """
        output_path = Path(output_path)
        html = self.render_leaderboard(leaderboard, title)
        output_path.write_text(html, encoding="utf-8")
        return output_path

    def save_calibration_report(
        self,
        result: CalibrationResultData,
        output_path: str | Path,
    ) -> Path:
        """Save calibration report HTML to file.

        Args:
            result: Calibration result data.
            output_path: Output file path.

        Returns:
            Path to saved file.
        """
        output_path = Path(output_path)
        html = self.render_calibration_report(result)
        output_path.write_text(html, encoding="utf-8")
        return output_path


# ============================================================================
# Convenience Functions
# ============================================================================


def generate_leaderboard_html(
    leaderboard: LeaderboardGenerator,
    output_path: str | Path | None = None,
    title: str = "UkrQualBench Leaderboard",
) -> str:
    """Generate leaderboard HTML.

    Args:
        leaderboard: Leaderboard generator.
        output_path: Optional path to save HTML file.
        title: Page title.

    Returns:
        HTML string.
    """
    generator = HTMLReportGenerator()
    html = generator.render_leaderboard(leaderboard, title)

    if output_path:
        Path(output_path).write_text(html, encoding="utf-8")

    return html


def generate_calibration_html(
    result: CalibrationResultData,
    output_path: str | Path | None = None,
) -> str:
    """Generate calibration report HTML.

    Args:
        result: Calibration result data.
        output_path: Optional path to save HTML file.

    Returns:
        HTML string.
    """
    generator = HTMLReportGenerator()
    html = generator.render_calibration_report(result)

    if output_path:
        Path(output_path).write_text(html, encoding="utf-8")

    return html
