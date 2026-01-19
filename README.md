# UkrQualBench

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**Benchmark for evaluating Ukrainian language quality in Large Language Models (LLMs)**

UkrQualBench evaluates **linguistic quality** (nativeness) of Ukrainian language in LLMs, not cognitive abilities. Unlike existing benchmarks measuring knowledge and reasoning, UkrQualBench focuses on how naturally and correctly models use Ukrainian.

---

## Leaderboard (January 2026)

### ELO Rating (Pairwise Comparisons)

| Rank | Model | ELO Rating | Badge |
|:----:|-------|:----------:|:-----:|
| 1 | **gpt-5.2-2025-12-11** | **1726** | 🥇 |
| 2 | claude-opus-4-5 | 1722 | 🥇 |
| 3 | gemini-3-flash-preview | 1681 | 🥇 |
| 4 | gemini-3-pro-preview | 1593 | 🥈 |
| 5 | claude-sonnet-4-5 | 1553 | 🥈 |
| 6 | mamaylm-gemma-3-12b-it-v1.0 | 1503 | 🥈 |
| 7 | google/gemma-3-27b-it-fast | 1422 | ⚠️ |
| 8 | lapa-v0.1.2-instruct | 1393 | ⚠️ |
| 9 | claude-haiku-4-5 | 1367 | ⚠️ |
| 10 | openai/gpt-oss-20b | 1305 | 🚫 |
| 11 | Qwen/Qwen3-32B-fast | 1236 | 🚫 |

### Block A: Calibration Tests (Absolute Scores)

| Model | MC Accuracy | GEC F1 | Translation | FP Rate | PM Score |
|-------|:-----------:|:------:|:-----------:|:-------:|:--------:|
| gpt-5.2-2025-12-11 | **0.93** | 0.77 | 0.90 | 0.0 | 0.8 |
| gemini-3-pro-preview | 0.89 | **0.84** | **0.97** | 0.4 | **0.9** |
| claude-opus-4-5 | 0.90 | 0.73 | 0.96 | 0.0 | 0.0 |
| gemini-3-flash-preview | 0.90 | 0.77 | 0.96 | 0.2 | 0.0 |
| google/gemma-3-27b-it-fast | 0.83 | 0.78 | **0.97** | 0.0 | 0.0 |
| Qwen/Qwen3-32B-fast | 0.78 | 0.70 | 0.77 | 0.1 | **0.9** |
| gpt-5-nano | 0.75 | 0.60 | 0.92 | 0.0 | 0.0 |
| mamaylm-gemma-3-12b-it-v1.0 | 0.74 | 0.64 | **0.97** | 0.0 | 0.8 |
| claude-sonnet-4-5 | 0.67 | 0.75 | 0.96 | 0.1 | 0.0 |
| claude-haiku-4-5 | 0.62 | 0.74 | 0.92 | 0.2 | 0.0 |
| openai/gpt-oss-20b | 0.60 | 0.61 | 0.80 | 0.0 | **0.9** |
| lapa-v0.1.2-instruct | 0.54 | 0.71 | 0.90 | 0.0 | 0.0 |

### Block V: Automatic Metrics (Detectors)

| Model | Fertility | Positive Markers | Calques* | Anglicisms |
|-------|:---------:|:----------------:|:--------:|:----------:|
| Qwen/Qwen3-32B-fast | 1.47 | **13.7** | 3.9 | 0.0 |
| gemini-3-flash-preview | 1.44 | 13.0 | 2.3 | 0.0 |
| claude-haiku-4-5 | 1.43 | 9.9 | 1.7 | 0.0 |
| gemini-3-pro-preview | 1.46 | 9.3 | 1.6 | 0.0 |
| mamaylm-gemma-3-12b-it-v1.0 | **1.40** | 7.3 | 1.6 | 0.0 |
| gpt-5.2-2025-12-11 | 1.43 | 6.9 | 6.0 | 0.0 |
| claude-sonnet-4-5 | 1.41 | 6.3 | 1.8 | 0.0 |
| claude-opus-4-5 | 1.49 | 6.0 | **0.0** | 0.0 |
| lapa-v0.1.2-instruct | **1.40** | 4.8 | 7.0 | 0.0 |
| google/gemma-3-27b-it-fast | 1.41 | 4.5 | 1.0 | 0.0 |
| openai/gpt-oss-20b | 1.50 | 4.4 | 1.9 | 0.0 |

*\*Calques = Russian calques detected by LLM judge (lexical, syntactic, morphological)*
**Note:** mamaylm-gemma-3-12b-it-v1.0 and lapa-v0.1.2-instruct were used with quantization (Q4_K_S)

<details>
<summary><b>Metrics Explanation</b></summary>

- **ELO Rating**: Swiss-system tournament rating (baseline 1500, K=32)
- **MC Accuracy**: Multiple choice accuracy (orthography, punctuation, russisms)
- **GEC F1**: Grammar error correction quality
- **Translation**: RU→UK translation quality score
- **FP Rate**: False positive rate (incorrectly "fixing" correct text)
- **PM Score**: Positive markers test score (vocative case, particles)
- **Fertility**: Tokens per word ratio (optimal ~1.4-1.5 for Ukrainian)
- **Positive Markers**: Native markers per 1000 tokens (higher = more natural)
- **Calques**: Russian calques per 1000 tokens detected by LLM judge (lower = better)
- **Anglicisms**: English calques per 1000 tokens (lower = better)

</details>

---

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

### Block V: Objective Metrics (automatic)
- Fertility rate (tokens/word ratio)
- Positive markers detection (regex-based)
- Calque detection (LLM judge-based for lexical, syntactic, morphological calques)
- Anglicism auto-detection (regex-based)

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

## Methodology

### What We Measure

UkrQualBench evaluates **linguistic nativeness**, not cognitive abilities:

| Aspect | What We Check | Example |
|--------|--------------|---------|
| **Russisms** | Calques from Russian | "прийняти участь" → "взяти участь" |
| **Orthography** | Correct spelling rules | "пів'яблука" vs "півяблука" |
| **Vocative Case** | Native address forms | "Пане Андрію" vs "Пан Андрій" |
| **Particles** | Ukrainian expressiveness | же, бо, адже, хіба, невже |
| **False Corrections** | Not "fixing" correct text | Classic literature should stay unchanged |

### Test Examples

<details>
<summary><b>Multiple Choice (Block A1)</b></summary>

```
Виберіть правильний варіант:
A) пів'яблука  ← correct (apostrophe before я)
B) півяблука
C) пів яблука

Яке слово є русизмом?
A) захід
B) міроприємство  ← russism (correct: захід)
C) подія
```

</details>

<details>
<summary><b>Grammar Error Correction (Block A2)</b></summary>

```
Input:  "Треба прийняти участь у заході."
Output: "Треба взяти участь у заході."
        ↑ "прийняти участь" is a russism

Input:  "На протязі року ми працювали."
Output: "Протягом року ми працювали."
        ↑ "на протязі" is a calque from Russian
```

</details>

<details>
<summary><b>Positive Markers Detection (Block V)</b></summary>

```
Good: "Друже, як справи? Адже ми ж домовлялися!"
       ↑       ↑        ↑    ↑
    vocative particle particle particle

Bad:  "Друг, как дела? Мы ведь договаривались!"
      (no Ukrainian markers, sounds translated)
```

</details>

### Critical Russisms to Detect

| Russism | Correct Form | Severity |
|---------|--------------|----------|
| прийняти участь | взяти участь | Critical |
| міроприємство | захід | Critical |
| на протязі | протягом | Critical |
| являється | є | Critical |
| слідуючий | наступний | Critical |
| отримати досвід | здобути досвід | High |

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

## Key Findings

### Model Comparison Insights

1. **GPT-5.2 and Claude Opus** tied at top ELO (~1720) with best MC accuracy (90-93%)
2. **Claude Opus 4.5** is the only model with **zero calques** detected — cleanest Ukrainian
3. **Gemini-3-Pro** has best GEC performance (84% F1) and highest PM score (0.9)
4. **Qwen3/Gemini-Flash** generate most "native" Ukrainian (13+ positive markers/1K tokens)
5. **GPT-5.2 and Lapa** have highest calque rates (6-7 per 1K), despite strong ELO scores
6. **Open models** (Gemma, MamaLyM) competitive with proprietary in linguistic quality

### Correlation Analysis

```
ELO Rating strongly correlates with MC Accuracy (r ≈ 0.7)
ELO Rating moderately correlates with GEC F1 (r ≈ 0.4)
Positive Markers inversely correlate with model size (smaller models use more native forms)
```

### Observations

- **Claude Opus 4.5** is the only model with **0.0 calque rate** — cleanest Ukrainian among tested models
- **GPT-5.2** and **Lapa** show highest calque rates (6.0-7.0 per 1K tokens), indicating more Russian influence
- **Fertility rate** is consistent across models (~1.4-1.5), indicating similar tokenization efficiency
- **Qwen3** has most positive markers (13.7/1K) but also elevated calque rate (3.9), suggesting mixed quality
- **Claude models** have lower positive markers than Gemini, suggesting more "formal" language style

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
