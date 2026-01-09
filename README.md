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

All commands should be run with `uv run` prefix:

```bash
# Show configuration and API key status
uv run ukrqualbench info

# Calibrate a judge model
uv run ukrqualbench calibrate --judge claude-3-5-haiku-latest

# Evaluate a single model (lite benchmark ~30min)
uv run ukrqualbench evaluate --model gpt-5.2 --benchmark lite

# Compare multiple models
uv run ukrqualbench compare --models gpt-5.2,claude-opus-4-5-20251101 --benchmark base

# Generate leaderboard
uv run ukrqualbench leaderboard --results-dir results/ --format html
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `info` | Show configuration, API keys, and benchmark sizes |
| `calibrate` | Calibrate a judge model against gold standards |
| `evaluate` | Evaluate a single model on the benchmark |
| `compare` | Compare multiple models using Swiss-system tournament |
| `leaderboard` | Generate leaderboard from evaluation results |

### Command Options

```bash
# Calibrate with custom output
uv run ukrqualbench calibrate --judge claude-3-5-haiku-latest --output results/calibration --verbose

# Evaluate with budget limit
uv run ukrqualbench evaluate --model gpt-5.2 --benchmark lite --max-cost 10.0 --resume

# Compare with specific round count
uv run ukrqualbench compare --models gpt-5.2,gemini-3-flash-preview --rounds 5 --judge claude-3-5-haiku-latest

# Leaderboard in different formats
uv run ukrqualbench leaderboard --results-dir results/ --format json  # or csv, markdown, html
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

## Supported Models

### Cloud Providers
| Provider | Models | Env Variable |
|----------|--------|--------------|
| **OpenAI** | gpt-5.2, gpt-5.2-pro, gpt-5-mini | `UKRQUALBENCH_OPENAI_API_KEY` |
| **Anthropic** | claude-opus-4-5-*, claude-sonnet-4-*, claude-haiku-4 | `UKRQUALBENCH_ANTHROPIC_API_KEY` |
| **Google** | gemini-3-pro-preview, gemini-3-flash-preview | `UKRQUALBENCH_GOOGLE_API_KEY` |
| **Nebius** | deepseek-ai/DeepSeek-R1, Qwen/*, meta-llama/* | `UKRQUALBENCH_NEBIUS_API_KEY` |

### Local Models
| Provider | Configuration |
|----------|---------------|
| **Ollama** | `UKRQUALBENCH_OLLAMA_BASE_URL` (default: http://localhost:11434) |
| **vLLM** | `UKRQUALBENCH_VLLM_BASE_URL` (default: http://localhost:8000) |

## Judge Calibration

Before using a judge model, it must pass calibration:

| Metric | Threshold |
|--------|-----------|
| MC Accuracy | > 85% |
| GEC F1 | > 80% |
| Russism Detection F1 | > 85% |
| False Positive Rate | < 15% |
| Pairwise Consistency | > 90% |
| Position Bias | < 5% |
| Length Bias | |r| < 0.30 |
| **Final Score** | > 0.80 |

## Quality Badges

| Badge | ELO | Russism Rate | Positive Markers | Fertility |
|-------|-----|--------------|------------------|-----------|
| 🥇 **Gold** | ≥ 1650 | < 1.0 | ≥ 5.0 | < 1.5 |
| 🥈 **Silver** | ≥ 1550 | < 3.0 | ≥ 3.0 | < 1.8 |
| 🥉 **Bronze** | ≥ 1450 | < 5.0 | ≥ 1.0 | < 2.0 |
| ⚠️ **Caution** | ≥ 1350 | < 10.0 | ≥ 0.0 | < 2.5 |
| 🚫 **Not Recommended** | < 1350 | ≥ 10.0 | — | — |

## Configuration

Configuration via environment variables (prefix: `UKRQUALBENCH_`):

```bash
# API Keys
UKRQUALBENCH_OPENAI_API_KEY=sk-...
UKRQUALBENCH_ANTHROPIC_API_KEY=sk-ant-...
UKRQUALBENCH_GOOGLE_API_KEY=...
UKRQUALBENCH_NEBIUS_API_KEY=...

# Settings
UKRQUALBENCH_BENCHMARK_VERSION=base
UKRQUALBENCH_DEFAULT_JUDGE=claude-3-5-haiku-latest
UKRQUALBENCH_MAX_COST_USD=50.0
UKRQUALBENCH_TEMPERATURE=0.0

# ELO Settings
UKRQUALBENCH_ELO_INITIAL_RATING=1500
UKRQUALBENCH_ELO_K_FACTOR=32

# Execution
UKRQUALBENCH_MAX_CONCURRENT_REQUESTS=10
UKRQUALBENCH_REQUEST_TIMEOUT=60
UKRQUALBENCH_CHECKPOINT_INTERVAL=100
```

See `.env.example` for all options.

## Data Sources

- **UA-GEC 2.0**: Grammar error correction (CC BY 4.0)
- **ZNO Dataset**: Multiple choice from Ukrainian standardized tests (MIT)
- **FLORES-200**: Translation benchmark (CC BY-SA 4.0)
- **Brown-UK**: Validation corpus (CC BY 4.0)

## Project Structure

```
ukrqualbench/
├── src/ukrqualbench/
│   ├── cli.py              # Command-line interface
│   ├── core/               # Evaluator, ELO, schemas, config
│   ├── datasets/           # Data loaders (UA-GEC, ZNO, FLORES, Brown-UK)
│   ├── detectors/          # Russism, anglicism, markers, fertility
│   ├── judges/             # LLM judge system, calibration
│   ├── models/             # API clients (OpenAI, Anthropic, Google, Nebius, local)
│   └── reports/            # Leaderboard, HTML, analysis
├── data/
│   ├── benchmarks/         # lite.json, base.json, large.json
│   ├── gold/               # Calibration datasets
│   └── dictionaries/       # Russism/anglicism patterns
└── tests/                  # 382 tests
```

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
