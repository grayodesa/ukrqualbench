"""Measure a model's fertility rate on the reference corpus."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import TYPE_CHECKING

from ukrqualbench.detectors.fertility import FertilityCalculator
from ukrqualbench.tokenization.corpus import WORD_PATTERN, load_reference_corpus
from ukrqualbench.tokenization.counters import ModelNotFoundError, TokenCounter, make_counter
from ukrqualbench.tokenization.registry import FertilityEntry
from ukrqualbench.tokenization.specs import get_spec

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ukrqualbench.core.config import Config
    from ukrqualbench.tokenization.corpus import ReferenceCorpus

logger = logging.getLogger(__name__)

DEFAULT_CONCURRENCY = 4


class UnknownModelError(KeyError):
    """The model has no tokenizer spec."""


async def count_documents(
    counter: TokenCounter, texts: Sequence[str], concurrency: int = DEFAULT_CONCURRENCY
) -> list[int]:
    """Count tokens of every text, at most ``concurrency`` requests at a time."""
    semaphore = asyncio.Semaphore(concurrency)

    async def one(text: str) -> int:
        async with semaphore:
            return await counter.count(text)

    return list(await asyncio.gather(*(one(text) for text in texts)))


def build_entry(
    model_id: str,
    counter: TokenCounter,
    counts: Sequence[int],
    corpus: ReferenceCorpus,
    note: str = "",
) -> FertilityEntry:
    """Aggregate per-document counts into a registry entry."""
    results = FertilityCalculator().calculate_batch(list(corpus.texts), list(counts))
    words = sum(r.word_count for r in results)
    if words != corpus.words:
        raise RuntimeError(f"word count mismatch: {words} != corpus {corpus.words}")
    tokens = sum(counts)
    return FertilityEntry(
        model_id=model_id,
        tokens=tokens,
        fertility_rate=round(tokens / corpus.words, 6),
        chars_per_token=round(corpus.chars / tokens, 6),
        corpus_sha256=corpus.sha256,
        word_pattern=WORD_PATTERN.pattern,
        method=counter.method,
        tokenizer_sha256=counter.tokenizer_sha256,
        baseline_overhead=counter.baseline_overhead,
        measured_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        note=note,
    )


async def measure_model(
    model_id: str,
    config: Config,
    *,
    corpus: ReferenceCorpus | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> FertilityEntry:
    """Count the reference corpus with the model's native tokenizer.

    Falls back to ``spec.fallback`` when the API reports the model as not found.

    Raises:
        UnknownModelError: No tokenizer spec for ``model_id``.
        ModelNotFoundError: API does not know the model and there is no fallback.
        CalibrationError: API wrapper overhead is not constant for this model.
    """
    spec = get_spec(model_id)
    if spec is None:
        raise UnknownModelError(f"no tokenizer spec for {model_id}")
    corpus = corpus or load_reference_corpus()

    counter = make_counter(spec, config)
    note = spec.note
    try:
        counts = await count_documents(counter, corpus.texts, concurrency)
    except ModelNotFoundError as e:
        if spec.fallback is None:
            raise
        logger.warning("%s: %s; using fallback %s", model_id, e, spec.fallback.ref)
        await counter.aclose()
        counter = make_counter(spec.fallback, config, is_fallback=True)
        counts = await count_documents(counter, corpus.texts, concurrency)
        note = f"{note}; API reported model not found, counted with fallback".lstrip("; ")
    finally:
        await counter.aclose()
    return build_entry(model_id, counter, counts, corpus, note)
