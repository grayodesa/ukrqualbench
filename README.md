# UkrQualBench

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**Benchmark for evaluating Ukrainian language quality in Large Language Models (LLMs)**

UkrQualBench evaluates **linguistic quality** (nativeness) of Ukrainian language in LLMs, not cognitive abilities. Unlike existing benchmarks measuring knowledge and reasoning, UkrQualBench focuses on how naturally and correctly models use Ukrainian.

## Key Principles

| Principle | Description |
|-----------|-------------|
| **Pairwise over Absolute** | Compare models against each other, not absolute scores |
| **Positive over Negative** | Reward native markers, not just penalize errors |
| **Real over Synthetic** | Real corpus data over synthetic examples |
| **Calibrated Judges** | LLM judges calibrated against gold standard |
| **Reproducible** | Deterministic results with temperature=0 |

## Installation

```bash
# Using uv (recommended)
uv sync

# With development dependencies
uv sync --extra dev

# With local model support
uv sync --extra local
```

## Quick Start

```bash
# Show configuration
ukrqualbench info

# Calibrate a judge model
ukrqualbench calibrate --judge claude-3-5-haiku-latest

# Evaluate a single model (lite benchmark ~30min)
ukrqualbench evaluate --model gpt-4o --benchmark lite

# Compare multiple models
ukrqualbench compare --models gpt-4o,claude-3.5-sonnet --benchmark base

# Generate leaderboard
ukrqualbench leaderboard --results-dir results/ --output leaderboard.html
```

## Benchmark Architecture

### Block A: Calibration Tests (with reference answers)
- **A1**: Multiple Choice (400 tasks) — ZNO Ukrainian, error detection
- **A2**: GEC (400 tasks) — Grammar error correction from UA-GEC
- **A3**: Translation (200 tasks) — EN→UK and RU→UK with COMET scoring
- **A4**: False Positives (50 tasks) — Ensure judges don't "correct" valid classic literature
- **A5**: Positive Markers (50 tasks) — Test for native language markers

### Block B: Generation Tests (pairwise evaluation)
- **B1**: Free generation (300 prompts) — explanations, advice, creative, technical
- **B2**: Adversarial (100 prompts) — test resistance to mimicking bad Ukrainian
- **B3**: Long context (50 prompts) — check language degradation over long contexts

### Block V: Objective Metrics (automatic, no judge)
- Fertility rate (tokens/word ratio)
- Positive markers detection (regex-based)
- Russism/anglicism auto-detection

## Benchmark Versions

| Version | Block A | Block B | Est. Time | Use Case |
|---------|---------|---------|-----------|----------|
| **lite** | 200 | 100 | ~30 min | Quick screening |
| **base** | 550 | 250 | ~2 hr | Standard evaluation |
| **large** | 1100 | 450 | ~5 hr | Full research |

## Judge Calibration

Before using a judge model, it must pass calibration:

| Metric | Threshold |
|--------|-----------|
| MC Agreement | > 85% |
| GEC F1 | > 80% |
| Russism Detection F1 | > 85% |
| False Positive Rate | < 15% |
| Pairwise Consistency | > 90% |
| **Final Score** | > 0.80 |

## Quality Badges

| Badge | Criteria |
|-------|----------|
| 🥇 **Gold** | ELO > 1700, russism_rate < 1.0 |
| 🥈 **Silver** | ELO > 1550, russism_rate < 3.0 |
| 🥉 **Bronze** | ELO > 1400, russism_rate < 5.0 |
| ⚠️ **Caution** | russism_rate > 10.0 |
| 🚫 **Not Recommended** | ELO < 1300 or russism_rate > 20.0 |

## Configuration

Configuration via environment variables (prefix: `UKRQUALBENCH_`):

```bash
# API Keys
UKRQUALBENCH_OPENAI_API_KEY=sk-...
UKRQUALBENCH_ANTHROPIC_API_KEY=sk-ant-...
UKRQUALBENCH_GOOGLE_API_KEY=...

# Settings
UKRQUALBENCH_BENCHMARK_VERSION=base
UKRQUALBENCH_DEFAULT_JUDGE=claude-3-5-haiku-latest
UKRQUALBENCH_MAX_COST_USD=50.0
```

See `.env.example` for all options.

## Data Sources

- **UA-GEC 2.0**: Grammar error correction (CC BY 4.0)
- **ZNO Dataset**: Multiple choice from Ukrainian standardized tests (MIT)
- **FLORES-200**: Translation benchmark (CC BY-SA 4.0)
- **Brown-UK**: Validation corpus (CC BY 4.0)

## Development

```bash
# Install dev dependencies
uv sync --extra dev

# Run tests
uv run pytest

# Run with coverage
uv run pytest --cov=ukrqualbench

# Linting
uv run ruff check .
uv run ruff format .

# Type checking
uv run mypy src/
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Citation

If you use UkrQualBench in your research, please cite:

```bibtex
@software{ukrqualbench2026,
  title = {UkrQualBench: Benchmark for Ukrainian Language Quality in LLMs},
  year = {2026},
  url = {https://github.com/ukrqualbench/ukrqualbench}
}
```
