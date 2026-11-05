"""Tests for native-tokenizer fertility: corpus, counters, registry, integration."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from ukrqualbench.core.schemas import (
    Badge,
    BlockAScores,
    BlockBScores,
    BlockVScores,
    ModelEvaluationData,
    ModelScoreData,
)
from ukrqualbench.detectors import RussismDetector
from ukrqualbench.reports.leaderboard import assign_badge
from ukrqualbench.tokenization import measure as measure_module
from ukrqualbench.tokenization.corpus import (
    WORD_PATTERN,
    ReferenceCorpus,
    corpus_fingerprint,
    count_words,
    load_reference_corpus,
)
from ukrqualbench.tokenization.counters import (
    HFTokenCounter,
    ModelNotFoundError,
    TiktokenCounter,
    make_counter,
)
from ukrqualbench.tokenization.measure import UnknownModelError, build_entry, measure_model
from ukrqualbench.tokenization.registry import (
    FertilityEntry,
    FertilityRegistry,
    fertility_block_v_fields,
)
from ukrqualbench.tokenization.specs import MODEL_TOKENIZERS, TokenizerSpec

if TYPE_CHECKING:
    from ukrqualbench.core.config import Config

FIXTURES = Path(__file__).parent / "fixtures"

# Expected values from the offline probe recorded in the plan (2026-11-05),
# computed before this implementation with a separate script.
CORPUS_SHA256 = "d9a80d3a4e9cd76f9d7a1724109cd826b786a3aa84296f9d703928c2c11e20d0"
CORPUS_WORDS = 35122
CORPUS_CHARS = 238096
LAPA_TOKENS = 53591
O200K_TOKENS = 83593

# Every model_id in results/evaluations/ on 2026-11-05 (results/ is not in git).
EVALUATED_MODELS = {
    "claude-fable-5",
    "claude-haiku-4-5",
    "claude-opus-4-5",
    "claude-opus-4-7",
    "claude-opus-5",
    "claude-opus-5-5",
    "claude-sonnet-4-5",
    "claude-sonnet-5",
    "claude-sonnet-5-5",
    "gemini-3-flash-preview",
    "gemini-3-pro-preview",
    "gemini-3.1-flash-lite-preview",
    "gemini-3.5-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "google/gemma-3-27b-it-fast",
    "google/gemma-4-26b-a4b",
    "gpt-5-nano",
    "gpt-5.2",
    "gpt-5.5",
    "gpt-5.6-sol",
    "gpt-6-astra",
    "gpt-6.1-sol",
    "lapa-v0.1.2-instruct",
    "mamaylm-gemma-3-12b-it-v1.0",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "openrouter:deepseek/deepseek-v4-flash",
    "openrouter:google/gemma-4-31b-it",
    "openrouter:meta/muse-glimmer-30b",
    "openrouter:qwen/qwen3.6-27b",
    "Qwen/Qwen3-32B-fast",
}


def _small_corpus() -> ReferenceCorpus:
    texts = ("Це текст.", "Ще один текст тут.")
    return ReferenceCorpus(
        corpus_id="test",
        doc_ids=("a", "b"),
        texts=texts,
        sha256=corpus_fingerprint(texts),
        words=sum(count_words(t) for t in texts),
        chars=sum(len(t) for t in texts),
    )


class FixedCounter:
    """Counter returning a fixed number of tokens per whitespace chunk."""

    def __init__(self, per_chunk: int = 2, method: str = "fake:counter") -> None:
        self._per_chunk = per_chunk
        self._method = method

    @property
    def method(self) -> str:
        return self._method

    @property
    def tokenizer_sha256(self) -> str | None:
        return None

    @property
    def baseline_overhead(self) -> int | None:
        return None

    async def count(self, text: str) -> int:
        return self._per_chunk * len(text.split())


class NotFoundCounter(FixedCounter):
    async def count(self, text: str) -> int:
        raise ModelNotFoundError("model retired")


def _entry(model_id: str, corpus_sha256: str = "f" * 64, **overrides: object) -> FertilityEntry:
    data: dict[str, object] = {
        "model_id": model_id,
        "tokens": 100,
        "fertility_rate": 2.5,
        "chars_per_token": 3.0,
        "corpus_sha256": corpus_sha256,
        "word_pattern": WORD_PATTERN.pattern,
        "method": "hf:org/repo@abcdef0",
        "tokenizer_sha256": None,
        "baseline_overhead": None,
        "measured_at": "2026-11-05T12:00:00+02:00",
        "note": "",
    }
    data.update(overrides)
    return FertilityEntry(**data)  # type: ignore[arg-type]


# ============================================================================
# Corpus and word counting
# ============================================================================


class TestReferenceCorpus:
    def test_corpus_matches_plan_fingerprint(self) -> None:
        corpus = load_reference_corpus()
        assert len(corpus.texts) == 166
        assert corpus.words == CORPUS_WORDS
        assert corpus.chars == CORPUS_CHARS
        assert corpus.sha256 == CORPUS_SHA256
        assert list(corpus.doc_ids) == sorted(corpus.doc_ids)

    @pytest.mark.parametrize("text", ["м'ята", "м’ята", "мʼята", "по-українськи", "2026"])
    def test_joined_word_is_one_word(self, text: str) -> None:
        assert WORD_PATTERN.findall(text) == [text]

    def test_lone_dash_is_not_a_word(self) -> None:
        assert WORD_PATTERN.findall(" - ") == []


# ============================================================================
# Tokenizer specs
# ============================================================================


class TestSpecs:
    def test_every_evaluated_model_has_a_spec(self) -> None:
        assert set(MODEL_TOKENIZERS) == EVALUATED_MODELS

    def test_hf_specs_are_pinned_to_full_commit(self) -> None:
        for model_id, spec in MODEL_TOKENIZERS.items():
            for s in (spec, spec.fallback):
                if s is not None and s.backend == "hf":
                    assert s.revision is not None and len(s.revision) == 40, model_id

    def test_only_pro_preview_has_fallback(self) -> None:
        with_fallback = {m for m, s in MODEL_TOKENIZERS.items() if s.fallback is not None}
        assert with_fallback == {"gemini-3-pro-preview"}


# ============================================================================
# Counters
# ============================================================================


class TestCounters:
    async def test_hf_counter_skips_special_tokens(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import WhitespaceSplit
        from tokenizers.processors import TemplateProcessing

        tok = Tokenizer(WordLevel({"[BOS]": 0, "a": 1, "[UNK]": 2}, unk_token="[UNK]"))
        tok.pre_tokenizer = WhitespaceSplit()
        tok.post_processor = TemplateProcessing(single="[BOS] $A", special_tokens=[("[BOS]", 0)])
        path = tmp_path / "tokenizer.json"
        tok.save(str(path))
        assert tok.encode("a a").ids == [0, 1, 1]  # the fixture does add BOS by default

        import huggingface_hub

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: str(path))
        counter = HFTokenCounter("org/repo", "0123456789abcdef")
        assert await counter.count("a a") == 2
        assert counter.method == "hf:org/repo@0123456"
        assert counter.tokenizer_sha256 is not None

    async def test_tiktoken_counter_ignores_special_token_text(self) -> None:
        counter = TiktokenCounter("o200k_base")
        assert await counter.count("<|endoftext|>") > 1
        assert counter.method == "tiktoken:o200k_base"

    def test_make_counter(self, config: Config) -> None:
        hf = make_counter(TokenizerSpec("hf", "org/repo", "0123456789abcdef"), config)
        assert hf.method.startswith("hf:")
        fallback = make_counter(
            TokenizerSpec("hf", "org/repo", "0123456789abcdef"), config, is_fallback=True
        )
        assert fallback.method.startswith("hf-fallback:")
        with pytest.raises(ValueError, match="no pinned revision"):
            make_counter(TokenizerSpec("hf", "org/repo"), config)
        with pytest.raises(NotImplementedError):
            make_counter(TokenizerSpec("anthropic", "claude-opus-5"), config)


# ============================================================================
# Registry
# ============================================================================


class TestRegistry:
    def test_round_trip(self, tmp_path: Path) -> None:
        corpus = _small_corpus()
        path = tmp_path / "fertility" / "registry.json"
        registry = FertilityRegistry(path, expected_corpus_sha256=corpus.sha256)
        entry = _entry("model-a", corpus.sha256)
        registry.put(entry)
        registry.save(corpus)

        loaded = FertilityRegistry.load(path, expected_corpus_sha256=corpus.sha256)
        assert loaded.get("model-a") == entry
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["schema_version"] == 1
        assert data["corpus"]["sha256"] == corpus.sha256

    def test_stale_corpus_is_not_returned(self, tmp_path: Path) -> None:
        corpus = _small_corpus()
        registry = FertilityRegistry(tmp_path / "r.json", expected_corpus_sha256=corpus.sha256)
        registry.put(_entry("model-a", "0" * 64))
        assert registry.get("model-a") is None
        assert "model-a" in registry.all_entries()

    def test_stale_word_pattern_is_not_returned(self, tmp_path: Path) -> None:
        corpus = _small_corpus()
        registry = FertilityRegistry(tmp_path / "r.json", expected_corpus_sha256=corpus.sha256)
        registry.put(_entry("model-a", corpus.sha256, word_pattern=r"[\w'-]+"))
        assert registry.get("model-a") is None

    def test_block_v_fields(self, tmp_path: Path) -> None:
        corpus = _small_corpus()
        registry = FertilityRegistry(tmp_path / "r.json", expected_corpus_sha256=corpus.sha256)
        registry.put(_entry("model-a", corpus.sha256))
        assert fertility_block_v_fields(registry, "model-a") == {
            "fertility_rate": 2.5,
            "fertility_method": "hf:org/repo@abcdef0",
        }
        assert fertility_block_v_fields(registry, "model-b") == {
            "fertility_rate": None,
            "fertility_method": None,
        }


# ============================================================================
# Measurement
# ============================================================================


class TestMeasure:
    def test_build_entry(self) -> None:
        corpus = _small_corpus()
        counts = [4, 8]
        entry = build_entry("model-a", FixedCounter(), counts, corpus, note="n")
        assert entry.tokens == 12
        assert entry.fertility_rate == round(12 / corpus.words, 6)
        assert entry.chars_per_token == round(corpus.chars / 12, 6)
        assert entry.corpus_sha256 == corpus.sha256
        assert entry.word_pattern == WORD_PATTERN.pattern
        assert entry.method == "fake:counter"
        assert entry.note == "n"

    async def test_unknown_model(self, config: Config) -> None:
        with pytest.raises(UnknownModelError):
            await measure_model("no-such-model", config, corpus=_small_corpus())

    async def test_fallback_on_model_not_found(
        self, config: Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fallback = TokenizerSpec("hf", "org/repo", "0123456789abcdef")
        spec = TokenizerSpec("gemini", "retired", fallback=fallback, note="retired ID")
        monkeypatch.setattr(measure_module, "get_spec", lambda model_id: spec)

        def fake_make_counter(s: TokenizerSpec, cfg: Config, *, is_fallback: bool = False):
            if is_fallback:
                return FixedCounter(method="hf-fallback:org/repo@0123456")
            return NotFoundCounter()

        monkeypatch.setattr(measure_module, "make_counter", fake_make_counter)
        entry = await measure_model("retired", config, corpus=_small_corpus())
        assert entry.method == "hf-fallback:org/repo@0123456"
        assert entry.note.startswith("retired ID; API reported model not found")

    async def test_not_found_without_fallback_raises(
        self, config: Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        spec = TokenizerSpec("gemini", "retired")
        monkeypatch.setattr(measure_module, "get_spec", lambda model_id: spec)
        monkeypatch.setattr(measure_module, "make_counter", lambda *a, **k: NotFoundCounter())
        with pytest.raises(ModelNotFoundError):
            await measure_model("retired", config, corpus=_small_corpus())

    @pytest.mark.integration
    @pytest.mark.parametrize(
        ("model_id", "expected_tokens"),
        [("lapa-v0.1.2-instruct", LAPA_TOKENS), ("openai/gpt-oss-20b", O200K_TOKENS)],
    )
    async def test_native_tokens_match_plan(
        self, config: Config, model_id: str, expected_tokens: int
    ) -> None:
        entry = await measure_model(model_id, config)
        assert entry.tokens == expected_tokens
        assert entry.fertility_rate == round(expected_tokens / CORPUS_WORDS, 6)
        assert entry.tokenizer_sha256 is not None


# ============================================================================
# Schema, badges, detector normalisation
# ============================================================================


def _scores(fertility: float | None) -> ModelScoreData:
    return ModelScoreData(
        elo_rating=1700,
        block_a=BlockAScores(
            mc_accuracy=0.9, gec_f1=0.8, translation_comet=0.9, false_positive_rate=0.1
        ),
        block_b=BlockBScores(),
        block_v=BlockVScores(
            fertility_rate=fertility,
            positive_markers=6.0,
            russism_rate=0.5,
            anglicism_rate=0.0,
        ),
    )


class TestIntegrationPoints:
    def test_old_evaluation_file_loads(self) -> None:
        data = json.loads((FIXTURES / "evaluation_v1.json").read_text(encoding="utf-8"))
        loaded = ModelEvaluationData.from_dict(data)
        assert loaded.block_v.fertility_method is None

        data["block_v"]["fertility_rate"] = None
        assert ModelEvaluationData.from_dict(data).block_v.fertility_rate is None

    @pytest.mark.parametrize("fertility", [3.4, None])
    def test_badge_ignores_fertility(self, fertility: float | None) -> None:
        assert assign_badge(_scores(fertility)) == Badge.GOLD

    def test_russism_rate_stays_on_pseudo_tokens(self) -> None:
        text = "Пане Іване, треба прийняти участь у мітингу. Бо ж дедлайн завтра!"
        result = RussismDetector().detect(text)
        assert result.total_tokens == max(1, int(len(text.split()) * 1.3))


class TestApplyCommand:
    def test_apply_writes_registry_values(self, tmp_path: Path) -> None:
        from ukrqualbench.cli import app

        evaluations = tmp_path / "evaluations"
        evaluations.mkdir()
        shutil.copy(FIXTURES / "evaluation_v1.json", evaluations / "claude-sonnet-5-5.json")
        other = json.loads((FIXTURES / "evaluation_v1.json").read_text(encoding="utf-8"))
        other["model_id"] = "unmeasured-model"
        (evaluations / "unmeasured-model.json").write_text(json.dumps(other), encoding="utf-8")

        corpus = load_reference_corpus()
        registry_path = tmp_path / "registry.json"
        registry = FertilityRegistry(registry_path, expected_corpus_sha256=corpus.sha256)
        registry.put(_entry("claude-sonnet-5-5", corpus.sha256, method="api:anthropic"))
        registry.save(corpus)

        result = CliRunner().invoke(
            app,
            [
                "fertility",
                "apply",
                "--evaluations-dir",
                str(evaluations),
                "--registry",
                str(registry_path),
            ],
        )
        assert result.exit_code == 0, result.output

        measured = json.loads((evaluations / "claude-sonnet-5-5.json").read_text())
        assert measured["block_v"]["fertility_rate"] == 2.5
        assert measured["block_v"]["fertility_method"] == "api:anthropic"
        unmeasured = json.loads((evaluations / "unmeasured-model.json").read_text())
        assert unmeasured["block_v"]["fertility_rate"] is None
        assert unmeasured["block_v"]["fertility_method"] is None
