"""Calibration of API token counts against independent references.

The wrapper-subtraction scheme of :class:`APITokenCounter` is a hypothesis.
These checks test it before any API count goes into the registry:

- exact match: where a local tokenizer of the same model exists, the API count
  of every reference document must equal the local count;
- split additivity: where no local reference exists (Anthropic), cutting a
  document in two must not change the total by more than a boundary merge.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ukrqualbench.tokenization.counters import CalibrationError, TokenCounter
from ukrqualbench.tokenization.measure import count_documents

if TYPE_CHECKING:
    from collections.abc import Sequence

SPLIT_DOCS = 20
SPLIT_TOLERANCE = 2


@dataclass(frozen=True)
class ExactMatchResult:
    """API counts versus a local tokenizer on the same documents."""

    label: str
    method: str
    reference: str
    docs: int
    mismatches: int
    diff_histogram: dict[int, int]
    overhead: int | None
    error: str = ""

    @property
    def passed(self) -> bool:
        return not self.error and self.mismatches == 0


@dataclass(frozen=True)
class SplitResult:
    """|n(A) + n(B) - n(A+B)| per document, API counter versus local reference."""

    label: str
    overhead: int | None
    gaps: list[int] = field(default_factory=list)
    reference_gaps: list[int] = field(default_factory=list)
    tolerance: int = SPLIT_TOLERANCE
    error: str = ""

    @property
    def max_gap(self) -> int:
        return max((abs(g) for g in self.gaps), default=0)

    @property
    def passed(self) -> bool:
        return not self.error and bool(self.gaps) and self.max_gap <= self.tolerance


def split_point(text: str) -> int:
    """Index of the space nearest to the middle of ``text`` (the middle if none)."""
    middle = len(text) // 2
    spaces = [i for i, ch in enumerate(text) if ch == " "]
    if not spaces:
        return middle
    return min(spaces, key=lambda i: (abs(i - middle), i))


async def exact_match(
    label: str,
    api: TokenCounter,
    reference: TokenCounter,
    texts: Sequence[str],
    concurrency: int = 4,
) -> ExactMatchResult:
    """Compare per-document API counts with a local tokenizer."""
    try:
        api_counts = await count_documents(api, texts, concurrency)
    except CalibrationError as e:
        return ExactMatchResult(
            label, api.method, reference.method, len(texts), 0, {}, None, str(e)
        )
    local_counts = await count_documents(reference, texts, concurrency)
    diffs = [a - b for a, b in zip(api_counts, local_counts, strict=True)]
    histogram = dict(Counter(d for d in diffs if d != 0).most_common(5))
    return ExactMatchResult(
        label=label,
        method=api.method,
        reference=reference.method,
        docs=len(texts),
        mismatches=sum(1 for d in diffs if d != 0),
        diff_histogram=histogram,
        overhead=api.baseline_overhead,
    )


async def _split_gaps(counter: TokenCounter, texts: Sequence[str]) -> list[int]:
    gaps = []
    for text in texts:
        i = split_point(text)
        whole, head, tail = (
            await counter.count(text),
            await counter.count(text[:i]),
            await counter.count(text[i:]),
        )
        gaps.append(head + tail - whole)
    return gaps


async def split_additivity(
    label: str,
    api: TokenCounter,
    reference: TokenCounter,
    texts: Sequence[str],
    docs: int = SPLIT_DOCS,
    tolerance: int = SPLIT_TOLERANCE,
) -> SplitResult:
    """Check that wrapper subtraction leaves only boundary effects on split documents."""
    sample = list(texts[:docs])
    try:
        gaps = await _split_gaps(api, sample)
    except CalibrationError as e:
        return SplitResult(label, None, tolerance=tolerance, error=str(e))
    reference_gaps = await _split_gaps(reference, sample)
    return SplitResult(label, api.baseline_overhead, gaps, reference_gaps, tolerance)


async def agreement_share(
    api: TokenCounter, reference: TokenCounter, texts: Sequence[str], concurrency: int = 4
) -> float:
    """Share of documents on which the API count equals the local count (informational)."""
    api_counts = await count_documents(api, texts, concurrency)
    local_counts = await count_documents(reference, texts, concurrency)
    equal = sum(1 for a, b in zip(api_counts, local_counts, strict=True) if a == b)
    return equal / len(texts) if texts else 0.0
