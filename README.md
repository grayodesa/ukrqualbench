# UkrQualBench

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**Benchmark for evaluating Ukrainian language quality in Large Language Models (LLMs)**

UkrQualBench evaluates **linguistic quality** (nativeness) of Ukrainian language in LLMs, not cognitive abilities. Unlike existing benchmarks measuring knowledge and reasoning, UkrQualBench focuses on how naturally and correctly models use Ukrainian.

---

## Leaderboard (November 2026)

### ELO Rating (Pairwise Comparisons)

| Rank | Model | ELO | MC Acc | GEC F1 | Fertility | Markers |
|------|-------|-----|--------|--------|-----------|---------|
| 1 | claude-opus-5-5 | 1882.0 | 94.0% | 75.9% | 2.96 | 11.1 |
| 2 | gemini-3.8-flash | 1869.4 | 90.0% | 80.4% | 2.26 | 10.0 |
| 3 | claude-opus-5 | 1825.7 | 90.0% | 82.1% | 2.96 | 8.1 |
| 4 | gemini-3.5-flash | 1712.2 | 86.0% | 80.3% | 2.26→ | 10.3 |
| 5 | gpt-6-astra | 1707.1 | 89.0% | 80.2% | 2.38 | 6.7 |
| 6 | gemini-3.1-flash-lite-preview | 1696.2 | 87.0% | 66.2% | 2.26 | 8.1 |
| 7 | gemini-3.7-flash | 1683.6 | 90.0% | 82.4% | 2.26→ | 7.3 |
| 8 | gemini-3-flash-preview | 1680.5 | 90.0% | 77.2% | 2.26 | 13.0 |
| 9 | claude-sonnet-5-5 | 1638.3 | 84.0% | 82.0% | 2.96 | 12.2 |
| 10 | claude-opus-4-7 | 1597.1 | 100.0% | 82.2% | 2.96 | 6.9 |
| 11 | gemini-3-pro-preview | 1593.0 | 89.1% | 84.2% | 2.26* | 9.3 |
| 12 | gpt-5.2 | 1572.5 | 93.0% | 76.7% | 2.38 | 6.9 |
| 13 | gpt-5.6-sol | 1572.1 | 82.0% | 78.7% | 2.38 | 8.0 |
| 14 | claude-opus-4-5 | 1565.7 | 90.0% | 73.0% | 2.94 | 6.0 |
| 15 | gpt-6.1-sol | 1560.6 | 89.0% | 82.2% | 2.38 | 7.4 |
| 16 | claude-fable-5 | 1553.7 | 89.0% | 86.2% | 2.96 | 8.7 |
| 17 | claude-sonnet-4-5 | 1553.3 | 67.0% | 75.0% | 2.94 | 6.3 |
| 18 | gpt-5.5 | 1518.7 | 82.0% | 73.7% | 2.38 | 8.3 |
| 19 | claude-haiku-5-5 | 1505.4 | 92.0% | 67.9% | 2.96 | 13.5 |
| 20 | google/gemma-4-26b-a4b | 1491.2 | 83.1% | 74.3% | 2.26 | 10.1 |
| 21 | openrouter:deepseek/deepseek-v4-flash | 1486.7 | 87.9% | 80.6% | 2.94 | 8.8 |
| 22 | google/gemma-3-27b-it-fast | 1421.8 | 83.0% | 78.4% | 2.26 | 4.5 |
| 23 | openrouter:meta/muse-glimmer-30b | 1417.8 | 89.9% | 76.3% | 2.15 | 7.4 |
| 24 | vllm:mamaylm-gemma-3-12b-it-v1.0 | 1404.1 | 66.0% | 78.3% | 2.26 | 15.7 |
| 25 | openrouter:google/gemma-4-31b-it | 1371.2 | 78.0% | 69.7% | 2.26 | 11.2 |
| 26 | vllm:mamaylm-gemma-3-27b-it-v2.0 | 1349.0 | 74.0% | 78.5% | 2.26 | 12.6 |
| 27 | gpt-5-nano | 1343.8 | 75.0% | 60.1% | 2.38 | 3.0 |
| 28 | claude-sonnet-5 | 1329.8 | 93.0% | 79.0% | 2.96 | 5.9 |
| 29 | openai/gpt-oss-20b | 1304.6 | 60.0% | 60.5% | 2.38 | 4.4 |
| 30 | vllm:mamaylm-gemma-3-12b-it-v2.0 | 1262.5 | 71.0% | 75.1% | 2.26 | 14.4 |
| 31 | Qwen/Qwen3-32B-fast | 1236.3 | 78.0% | 69.7% | 3.38 | 13.7 |
| 32 | claude-haiku-4-5 | 1172.2 | 62.0% | 74.0% | 2.94 | 9.9 |
| 33 | vllm:lapa-v0.1.2-instruct | 1159.5 | 58.0% | 68.4% | 1.53 | 5.5 |
| 34 | vllm:lapa-v0.1.3-instruct | 1066.6 | 61.0% | 68.4% | 1.53 | 5.2 |

Fertility: native-tokenizer tokens per word on the UA-GEC test corpus (35,122 words); lower is cheaper. Claude, GPT and Gemini are counted with the providers' token-count APIs (Anthropic counts are provider estimates). \* proxy tokenizer for a retired model ID (google-genai LocalTokenizer map). → the provider routes this ID to a newer model; the count reflects that model. Not shown: lapa-v0.1.2-instruct (Q4_K_S GGUF in LM Studio, January 2026; BF16 run: vllm:lapa-v0.1.2-instruct); mamaylm-gemma-3-12b-it-v1.0 (Q4_K_S GGUF in LM Studio, January 2026; BF16 run: vllm:mamaylm-gemma-3-12b-it-v1.0)

**Note:** `vllm:` rows are BF16 checkpoints served with vLLM 0.31 on RunPod (November 2026). The January runs of lapa-v0.1.2-instruct and mamaylm-gemma-3-12b-it-v1.0 used Q4_K_S GGUF in LM Studio; they stay in the ELO registry but are not shown, since quantization and the serving stack make them incomparable with the BF16 runs. The calibration tournaments of November 5 also moved the ratings of their anchor models (anchors are not frozen while new models are rated); a fix to the ELO calibration is planned.

<details>
<summary><b>Metrics Explanation</b></summary>

- **ELO Rating**: Swiss-system tournament rating (baseline 1500, K=32)
- **MC Acc**: Multiple choice accuracy (orthography, punctuation, russisms)
- **GEC F1**: Grammar error correction quality
- **Fertility**: Tokens per word that the model's own tokenizer spends on a fixed Ukrainian reference corpus (lower = cheaper, more text per context window). Describes the tokenizer, not the quality of the model's Ukrainian; see *Fertility Methodology*
- **Russisms** (column hidden): `evaluate` counts matches of a 28-pattern regex dictionary (`data/dictionaries/russisms.json`) in 10 generated answers. A manual audit of current model outputs (November 2026) found that it misses nearly all russisms these models actually produce, so the near-zero values do not rank models. The column returns after the dictionary is expanded and all models are re-measured.
- **Markers**: Native positive markers per 1000 pseudo-tokens (higher = more natural)

</details>

<details>
<summary><b>Fertility Methodology</b></summary>

- **Corpus**: the corrected (`target`) side of the UA-GEC 2.1.3 `gec-only` test split, shipped in the `ua-gec` package — 166 essays by native speakers, 35,122 words, 238,096 characters (SHA-256 `d9a80d3a…`). Every model is measured on the same text, so differences come from tokenizers only.
- **Words**: `\w+(?:['’ʼ-]\w+)*` — apostrophes (`'`, U+2019, U+02BC) and hyphens inside a word do not split it; a lone dash is not a word. Deviation from UA-GEC's gold tokenization: +0.07%.
- **Tokens**: native tokenizer of each model, no BOS/EOS or chat template, summed over documents. `FR = tokens / words`; `chars/token` is stored too.
- **Backends**:
  - open-weight models — official `tokenizer.json` from Hugging Face, pinned to a commit;
  - GPT — OpenAI `responses/input_tokens`; Gemini — `countTokens`; Claude — Anthropic `count_tokens`. These endpoints count a whole message, so the wrapper is measured from one-token texts (`.`, `a`, `x`) and subtracted.
- **Calibration** (`ukrqualbench fertility calibrate`): API counts match local tokenizers exactly where one exists (gpt-5-nano = o200k_base, gemini-3-flash-preview = Gemma 3, all 166 documents). For Claude there is no local tokenizer; in the 2026-11-05 run, splitting each of 20 documents in two changed its count by at most one token (the check allows two). Anthropic documents its count as an estimate.
- **Limitations**: absolute values depend on the corpus genre (essays) and are not comparable with fertility measured on other corpora, only between models here. For IDs the provider routes to a newer model (`→`) the count belongs to that model. The tokenizers of GPT 5.5+/6.x and Gemini 3.7/3.8 are not published; their API counts happen to equal o200k_base and Gemma 4 on every document of this corpus.
- **Markers**: `*` — the API no longer serves the model ID, counted with the tokenizer `google-genai` maps it to; `→` — the provider routes the ID to a newer model, so the count reflects that model.
- **Commands**: `ukrqualbench fertility measure --all`, `fertility show`, `fertility apply` (writes registry values into `results/evaluations/`). Values live in `data/fertility/registry.json`; install the tokenizers with `uv sync --extra fertility`.

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
uv run ukrqualbench calibrate --judge gemini-3.1-flash-lite

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
uv run ukrqualbench calibrate --judge gemini-3.1-flash-lite --output results/calibration --verbose

# Evaluate with budget limit
uv run ukrqualbench evaluate --model gpt-5.2 --benchmark lite --max-cost 10.0 --resume

# Compare with specific round count
uv run ukrqualbench compare --models gpt-5.2,gemini-3-flash-preview --rounds 5 --judge gemini-3.1-flash-lite

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
- Fertility rate (native tokenizer, fixed reference corpus)
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
| **Anthropic** | claude-fable-5, claude-opus-5, claude-sonnet-5, claude-opus-4-*, claude-sonnet-4-*, claude-haiku-4-5 | `UKRQUALBENCH_ANTHROPIC_API_KEY` |
| **Google** | gemini-3-pro-preview, gemini-3-flash-preview | `UKRQUALBENCH_GOOGLE_API_KEY` |

### Local Models
| Provider | Configuration |
|----------|---------------|
| **LM Studio** (and any OpenAI-compatible server) | `UKRQUALBENCH_LOCAL_BASE_URL` (default: http://localhost:1234/v1) |
| **Ollama** | `UKRQUALBENCH_OLLAMA_BASE_URL` (default: http://localhost:11434) |
| **vLLM** | `UKRQUALBENCH_VLLM_BASE_URL` (default: http://localhost:8000), `UKRQUALBENCH_VLLM_ENDPOINTS` (JSON map model → URL, for several vLLM servers at once), `UKRQUALBENCH_VLLM_API_KEY` (if the server runs with `--api-key`) |

Any model id containing `gemma` (e.g. `google/gemma-4-26b-a4b`) is routed to the local OpenAI-compatible endpoint — load the model in LM Studio and run `ukrqualbench evaluate --model google/gemma-4-26b-a4b`.

### Self-hosted models on RunPod

`scripts/runpod_pods.py` serves open-weight models on RunPod GPU pods (one pod per model, image `vllm/vllm-openai:v0.31.0`, HF revision pinned) and wires them into the `vllm:` provider. It reads `UKRQUALBENCH_RUNPOD_API_KEY` from `.env` and never prints keys.

```bash
uv run python scripts/runpod_pods.py keygen   # once: UKRQUALBENCH_VLLM_API_KEY for the vLLM servers
uv run python scripts/runpod_pods.py create   # pods for the models listed in the script; writes UKRQUALBENCH_VLLM_ENDPOINTS
uv run python scripts/runpod_pods.py status   # pod state and a /v1/models probe per model
uv run ukrqualbench evaluate -m vllm:lapa-v0.1.3-instruct -b lite
uv run python scripts/runpod_pods.py delete   # terminate the pods (billing is per second until then)
```

`create`, `switch` and `delete` accept model names to act on a subset; `switch OLD NEW` restarts a pod with another model. `compare` generates answers of all its models in every round, so all their pods must be up at once. Pods are started with `--max-model-len 16384`, because the client requests up to 8192 new tokens. RunPod's HTTP proxy closes a request after 100 seconds (HTTP 524), which cuts off very long answers.

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

| Badge | ELO | Russism Rate | Positive Markers |
|-------|-----|--------------|------------------|
| 🥇 **Gold** | ≥ 1650 | < 1.0 | ≥ 5.0 |
| 🥈 **Silver** | ≥ 1550 | < 3.0 | ≥ 3.0 |
| 🥉 **Bronze** | ≥ 1450 | < 5.0 | ≥ 1.0 |
| ⚠️ **Caution** | ≥ 1350 | < 10.0 | ≥ 0.0 |
| 🚫 **Not Recommended** | < 1350 | ≥ 10.0 | — |

Fertility is not a badge criterion: it measures the tokenizer's cost per word, not the quality of the model's Ukrainian.

## Configuration

Configuration via environment variables (prefix: `UKRQUALBENCH_`):

```bash
# API Keys
UKRQUALBENCH_OPENAI_API_KEY=sk-...
UKRQUALBENCH_ANTHROPIC_API_KEY=sk-ant-...
UKRQUALBENCH_GOOGLE_API_KEY=...

# Local models (LM Studio, Ollama, vLLM, etc.)
UKRQUALBENCH_LOCAL_BASE_URL=http://localhost:1234/v1

# Settings
UKRQUALBENCH_BENCHMARK_VERSION=base
UKRQUALBENCH_DEFAULT_JUDGE=gemini-3.1-flash-lite
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
│   ├── models/             # API clients (OpenAI, Anthropic, Google, local OpenAI-compatible)
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
- **Fertility rate** (native tokenizers, same corpus) spans 1.53–3.38 tokens per word: Lapa 1.53 (its vocabulary replaces Gemma's Cyrillic tokens with Ukrainian ones), Muse-Glimmer 2.15, Gemma 3/4 and Gemini 2.26, GPT (o200k) 2.38, Claude 2.94–2.96, DeepSeek V4 2.94, Qwen3 3.38. Models sharing a tokenizer share the value; the Claude 4.7+ tokenizer costs only ~1% more than the 4.5 one on Ukrainian
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
