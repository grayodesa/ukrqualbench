"""Fertility rate calculator for Ukrainian text.

Fertility rate is the ratio of tokens (subword units) to words. It describes a
tokenizer: how many tokens it spends on a Ukrainian word. Lower is cheaper and
leaves more context for the same text.

Token counts must come from a real tokenizer — a ``tokenizer`` with
``encode``, a ``token_counter`` callable, or precomputed counts. There is no
heuristic estimate: an invented token count is indistinguishable from a
measured one. The benchmark's per-model fertility is measured on a fixed
reference corpus by :mod:`ukrqualbench.tokenization`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from ukrqualbench.tokenization.corpus import WORD_PATTERN

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "WORD_PATTERN",
    "FertilityCalculator",
    "FertilityResult",
    "Tokenizer",
    "calculate_fertility",
    "evaluate_fertility_quality",
]

NO_TOKEN_SOURCE = "FertilityCalculator requires a tokenizer, token_counter or precomputed_tokens"

# Quality bands for native-tokenizer fertility on Ukrainian text. Measured on
# the UA-GEC reference corpus: Lapa 1.53 (Ukrainian vocabulary), Muse-Glimmer,
# Gemma and o200k 2.15-2.38, DeepSeek V4 2.94, Qwen3 3.38.
QUALITY_BANDS: tuple[tuple[float, str], ...] = (
    (1.8, "excellent"),
    (2.4, "good"),
    (3.0, "acceptable"),
    (3.5, "poor"),
)


class Tokenizer(Protocol):
    """Protocol for tokenizer interface."""

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        ...


def evaluate_fertility_quality(fertility_rate: float) -> str:
    """Quality band of a native-tokenizer fertility rate.

    Args:
        fertility_rate: Tokens per word.

    Returns:
        One of excellent, good, acceptable, poor, very_poor.
    """
    for upper, label in QUALITY_BANDS:
        if fertility_rate < upper:
            return label
    return "very_poor"


@dataclass
class FertilityResult:
    """Result of fertility rate calculation."""

    text: str
    word_count: int
    token_count: int
    fertility_rate: float
    metadata: dict[str, Any]

    @property
    def is_efficient(self) -> bool:
        """Whether the rate is in the excellent or good band (< 2.4)."""
        return self.quality_level in ("excellent", "good")

    @property
    def quality_level(self) -> str:
        """Quality band, see :func:`evaluate_fertility_quality`."""
        return evaluate_fertility_quality(self.fertility_rate)


class FertilityCalculator:
    """Calculator for text fertility rate (tokens / words).

    Example:
        >>> calc = FertilityCalculator(token_counter=lambda t: 6)
        >>> calc.calculate("Це простий текст.").fertility_rate
        2.0
    """

    def __init__(
        self,
        tokenizer: Tokenizer | None = None,
        token_counter: Callable[[str], int] | None = None,
    ) -> None:
        """Initialize calculator.

        Args:
            tokenizer: Optional tokenizer with an ``encode`` method.
            token_counter: Optional function returning the token count of a text.
        """
        self._tokenizer = tokenizer
        self._token_counter = token_counter

    def calculate(
        self,
        text: str,
        precomputed_tokens: int | None = None,
    ) -> FertilityResult:
        """Calculate fertility rate for text.

        Args:
            text: Text to analyze.
            precomputed_tokens: Token count computed elsewhere (optional).

        Returns:
            Fertility result with rate and metadata.

        Raises:
            ValueError: No tokenizer, counter or precomputed count is available.
        """
        word_count = len(WORD_PATTERN.findall(text))

        if precomputed_tokens is not None:
            token_count = precomputed_tokens
            method = "precomputed"
        else:
            token_count = self._count_tokens(text)
            method = self._get_estimation_method()

        fertility_rate = 0.0 if word_count == 0 else token_count / word_count

        return FertilityResult(
            text=text,
            word_count=word_count,
            token_count=token_count,
            fertility_rate=fertility_rate,
            metadata={"estimation_method": method},
        )

    def calculate_batch(
        self,
        texts: list[str],
        precomputed_tokens: list[int] | None = None,
    ) -> list[FertilityResult]:
        """Calculate fertility rate for multiple texts.

        Args:
            texts: List of texts to analyze.
            precomputed_tokens: Token counts aligned with ``texts`` (optional).

        Returns:
            List of fertility results.
        """
        if precomputed_tokens is None:
            return [self.calculate(text) for text in texts]
        return [
            self.calculate(text, tokens)
            for text, tokens in zip(texts, precomputed_tokens, strict=True)
        ]

    def calculate_aggregate(self, texts: list[str]) -> dict[str, float]:
        """Calculate aggregate fertility statistics.

        Args:
            texts: List of texts to analyze.

        Returns:
            Dictionary with aggregate statistics.
        """
        results = self.calculate_batch(texts)

        total_words = sum(r.word_count for r in results)
        total_tokens = sum(r.token_count for r in results)
        rates = [r.fertility_rate for r in results if r.fertility_rate > 0]

        aggregate_rate = total_tokens / total_words if total_words > 0 else 0.0

        return {
            "total_texts": len(texts),
            "total_words": total_words,
            "total_tokens": total_tokens,
            "aggregate_fertility_rate": aggregate_rate,
            "mean_fertility_rate": sum(rates) / len(rates) if rates else 0.0,
            "min_fertility_rate": min(rates) if rates else 0.0,
            "max_fertility_rate": max(rates) if rates else 0.0,
        }

    def compare_texts(
        self,
        text_a: str,
        text_b: str,
        tokens_a: int | None = None,
        tokens_b: int | None = None,
    ) -> dict[str, Any]:
        """Compare fertility rates between two texts.

        Args:
            text_a: First text.
            text_b: Second text.
            tokens_a: Pre-computed tokens for text A.
            tokens_b: Pre-computed tokens for text B.

        Returns:
            Comparison results.
        """
        result_a = self.calculate(text_a, tokens_a)
        result_b = self.calculate(text_b, tokens_b)

        diff = result_a.fertility_rate - result_b.fertility_rate

        return {
            "text_a_fertility": result_a.fertility_rate,
            "text_b_fertility": result_b.fertility_rate,
            "difference": diff,
            "more_efficient": "A" if diff < 0 else ("B" if diff > 0 else "equal"),
            "text_a_quality": result_a.quality_level,
            "text_b_quality": result_b.quality_level,
        }

    def _count_tokens(self, text: str) -> int:
        """Count tokens with the configured counter or tokenizer.

        Raises:
            ValueError: Neither a counter nor a tokenizer is configured.
        """
        if self._token_counter is not None:
            return self._token_counter(text)
        if self._tokenizer is not None:
            return len(self._tokenizer.encode(text))
        raise ValueError(NO_TOKEN_SOURCE)

    def _get_estimation_method(self) -> str:
        """Get the method used for token counting."""
        if self._token_counter is not None:
            return "custom_counter"
        if self._tokenizer is not None:
            return "tokenizer"
        raise ValueError(NO_TOKEN_SOURCE)


def calculate_fertility(text: str, token_count: int) -> float:
    """Fertility rate of ``text`` given its token count.

    Args:
        text: Text to analyze.
        token_count: Token count from a real tokenizer.

    Returns:
        Fertility rate (tokens / words).
    """
    return FertilityCalculator().calculate(text, token_count).fertility_rate
