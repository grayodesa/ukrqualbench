"""Which native tokenizer counts tokens for each benchmark model.

Keys are exact ``model_id`` values used in evaluation files. There is no
prefix matching: a model missing from :data:`MODEL_TOKENIZERS` has no
fertility value. Hugging Face tokenizers are pinned to commit SHAs (taken
2026-11-05) so the counts stay reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Backend = Literal["hf", "tiktoken", "anthropic", "openai", "gemini"]


@dataclass(frozen=True)
class TokenizerSpec:
    """How to count tokens for one model.

    Attributes:
        backend: Counting mechanism.
        ref: Hugging Face repo, tiktoken encoding name, or upstream API model id.
        revision: Commit SHA for ``hf`` specs.
        fallback: Spec used when the API reports the model as not found.
        note: Free-form remark copied into the registry entry.
    """

    backend: Backend
    ref: str
    revision: str | None = None
    fallback: TokenizerSpec | None = None
    note: str = ""


def _hf(repo: str, revision: str, note: str = "") -> TokenizerSpec:
    return TokenizerSpec(backend="hf", ref=repo, revision=revision, note=note)


def _api(
    backend: Backend, model: str, note: str = "", fallback: TokenizerSpec | None = None
) -> TokenizerSpec:
    return TokenizerSpec(backend=backend, ref=model, fallback=fallback, note=note)


# Gemma 3 tokenizer. google/gemma-3-*-it repos are gated; the MamayLM repo
# carries a byte-identical tokenizer.json (git blob 29401f98...).
GEMMA3 = _hf(
    "INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v1.0", "2d4fd7a65bef3534463a2abddeefff2c70fa0e83"
)
GEMMA4_31B = _hf("google/gemma-4-31B-it", "842da3794eaa0b77d5f08bae87a17459d91ff475")

# Reference tokenizers for calibrating API counts (plan, phase 2).
O200K_BASE = TokenizerSpec(backend="tiktoken", ref="o200k_base")

MODEL_TOKENIZERS: dict[str, TokenizerSpec] = {
    # Anthropic: tokenizer is not published; free count_tokens endpoint.
    "claude-opus-5": _api("anthropic", "claude-opus-5", "tokenizer generation 4.7+"),
    "claude-opus-5-5": _api("anthropic", "claude-opus-5-5", "tokenizer generation 4.7+"),
    "claude-sonnet-5-5": _api("anthropic", "claude-sonnet-5-5", "Sonnet 5 tokenizer"),
    "claude-sonnet-5": _api("anthropic", "claude-sonnet-5", "tokenizer generation 4.7+"),
    "claude-fable-5": _api("anthropic", "claude-fable-5", "tokenizer generation 4.7+"),
    "claude-opus-4-7": _api("anthropic", "claude-opus-4-7", "first model with the 4.7+ tokenizer"),
    "claude-opus-4-5": _api("anthropic", "claude-opus-4-5", "pre-4.7 tokenizer"),
    "claude-sonnet-4-5": _api(
        "anthropic", "claude-sonnet-4-5", "pre-4.7 tokenizer; EOL 2026-11-30"
    ),
    "claude-haiku-4-5": _api("anthropic", "claude-haiku-4-5", "pre-4.7 tokenizer"),
    # Google Gemini: free countTokens endpoint.
    "gemini-3.8-flash": _api("gemini", "gemini-3.8-flash"),
    "gemini-3.7-flash": _api("gemini", "gemini-3.7-flash", "ID routes to gemini-3.8-flash"),
    "gemini-3.5-flash": _api("gemini", "gemini-3.5-flash", "ID routes to gemini-3.6-flash"),
    "gemini-3-flash-preview": _api("gemini", "gemini-3-flash-preview"),
    "gemini-3-pro-preview": _api(
        "gemini",
        "gemini-3-pro-preview",
        "shutdown date 2026-03-09; fallback per google-genai LocalTokenizer map (gemma3)",
        fallback=GEMMA3,
    ),
    "gemini-3.1-flash-lite-preview": _api(
        "gemini",
        "gemini-3.1-flash-lite-preview",
        "shutdown date 2026-05-25; no verified local tokenizer for the preview ID",
    ),
    # OpenAI: responses/input_tokens endpoint.
    "gpt-6-astra": _api("openai", "gpt-6-astra"),
    "gpt-6.1-sol": _api("openai", "gpt-6.1-sol"),
    "gpt-5.6-sol": _api("openai", "gpt-5.6-sol"),
    "gpt-5.5": _api("openai", "gpt-5.5"),
    "gpt-5.2": _api("openai", "gpt-5.2"),
    "gpt-5-nano": _api("openai", "gpt-5-nano"),
    # Open weights: official tokenizer.json on Hugging Face.
    "lapa-v0.1.2-instruct": _hf(
        "lapa-llm/lapa-v0.1.2-instruct",
        "10116908ee519a816cfcb4b4e90668ba8e0710a8",
        "Gemma 3 vocabulary with Cyrillic tokens replaced",
    ),
    "mamaylm-gemma-3-12b-it-v1.0": GEMMA3,
    # Self-hosted on vLLM. tokenizer.json sha256 matches lapa-v0.1.2 (8332e5fa...)
    # and the Gemma 3 file above (4667f208...) respectively.
    "vllm:lapa-v0.1.3-instruct": _hf(
        "lapa-llm/lapa-v0.1.3-instruct",
        "66ba3ebfa6baa17bc2b3222f3908c30152ae104f",
        "lapa-v0.1.2 + merged DPO adapter; same tokenizer",
    ),
    "vllm:mamaylm-gemma-3-12b-it-v2.0": _hf(
        "INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v2.0", "3cb02013b6b37cfe10f4b3da0fcddc1acef7f244"
    ),
    "vllm:mamaylm-gemma-3-27b-it-v2.0": _hf(
        "INSAIT-Institute/MamayLM-Gemma-3-27B-IT-v2.0", "ad1bdfed9646939358adfcba9c88bb4031b50ed7"
    ),
    # Previous versions served in BF16 on vLLM; the unprefixed IDs above are Q4 GGUF runs.
    "vllm:lapa-v0.1.2-instruct": _hf(
        "lapa-llm/lapa-v0.1.2-instruct",
        "10116908ee519a816cfcb4b4e90668ba8e0710a8",
        "BF16 on vLLM; lapa-v0.1.2-instruct is the Q4 GGUF run",
    ),
    "vllm:mamaylm-gemma-3-12b-it-v1.0": GEMMA3,
    "vllm:mamaylm-gemma-3-27b-it-v2.0-fp8": _hf(
        "INSAIT-Institute/MamayLM-Gemma-3-27B-IT-v2.0-FP8-dynamic",
        "7ef42c6e4991864f0f47bfd524c1e77c9bdb11a9",
        "official FP8 build of the 27B; same tokenizer.json",
    ),
    "google/gemma-3-27b-it-fast": TokenizerSpec(
        backend="hf",
        ref=GEMMA3.ref,
        revision=GEMMA3.revision,
        note="google/gemma-3-27b-it is gated; identical tokenizer.json from the MamayLM repo",
    ),
    "google/gemma-4-26b-a4b": _hf(
        "google/gemma-4-26B-A4B-it", "4d7ae4984b7db7de8f8457170b3f1a419ee76d52"
    ),
    "openrouter:google/gemma-4-31b-it": GEMMA4_31B,
    "Qwen/Qwen3-32B-fast": _hf("Qwen/Qwen3-32B", "9216db5781bf21249d130ec9da846c4624c16137"),
    "openrouter:qwen/qwen3.6-27b": _hf(
        "Qwen/Qwen3.6-27B", "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
    ),
    "openai/gpt-oss-20b": _hf("openai/gpt-oss-20b", "6cee5e81ee83917806bbde320786a8fb61efebee"),
    "openai/gpt-oss-120b": _hf("openai/gpt-oss-120b", "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"),
    "openrouter:deepseek/deepseek-v4-flash": _hf(
        "deepseek-ai/DeepSeek-V4-Flash", "60d8d70770c6776ff598c94bb586a859a38244f1"
    ),
    "openrouter:meta/muse-glimmer-30b": _hf(
        "meta-models/Muse-Glimmer-30B", "cec94fcfd6512f2cfe708138c32533b346ea4f69"
    ),
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B": _hf(
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "bf77c3174f68ad409e1c2aa60daeb46e32d1c606"
    ),
}


def get_spec(model_id: str) -> TokenizerSpec | None:
    """Return the tokenizer spec for ``model_id``, or ``None`` if there is none."""
    return MODEL_TOKENIZERS.get(model_id)
