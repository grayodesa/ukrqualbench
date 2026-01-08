"""Fertility rate calculator for Ukrainian text.

Fertility rate measures the ratio of tokens (subword units) to words.
Lower fertility rates indicate:
- More efficient tokenization
- Better language representation in the model
- Native-like vocabulary usage

Ukrainian text typically has higher fertility rates than English
due to rich morphology and Cyrillic script considerations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable


# Word splitting pattern for Ukrainian text
WORD_PATTERN = re.compile(r"[\w''-]+", re.UNICODE)


class Tokenizer(Protocol):
    """Protocol for tokenizer interface."""

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        ...


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
        """Check if fertility rate indicates efficient tokenization.

        For Ukrainian, rates below 2.0 are considered efficient.
        """
        return self.fertility_rate < 2.0

    @property
    def quality_level(self) -> str:
        """Get quality level based on fertility rate.

        Returns:
            Quality level string.
        """
        if self.fertility_rate < 1.5:
            return "excellent"
        if self.fertility_rate < 2.0:
            return "good"
        if self.fertility_rate < 2.5:
            return "acceptable"
        if self.fertility_rate < 3.0:
            return "poor"
        return "very_poor"


class FertilityCalculator:
    """Calculator for text fertility rate.

    Fertility rate = tokens / words

    A lower rate indicates better tokenization efficiency.
    For Ukrainian text:
    - Excellent: < 1.5
    - Good: 1.5 - 2.0
    - Acceptable: 2.0 - 2.5
    - Poor: 2.5 - 3.0
    - Very poor: > 3.0

    Example:
        >>> calc = FertilityCalculator()
        >>> # Using simple estimation
        >>> result = calc.calculate("Це простий текст.")
        >>> result.fertility_rate
        1.3
    """

    # Default estimation factor for Ukrainian text
    # Based on typical tokenizer behavior
    DEFAULT_TOKEN_FACTOR = 1.3

    def __init__(
        self,
        tokenizer: Tokenizer | None = None,
        token_counter: Callable[[str], int] | None = None,
    ) -> None:
        """Initialize calculator.

        Args:
            tokenizer: Optional tokenizer with encode method.
            token_counter: Optional custom function to count tokens.
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
            precomputed_tokens: Pre-computed token count (optional).

        Returns:
            Fertility result with rate and metadata.
        """
        # Count words
        words = WORD_PATTERN.findall(text)
        word_count = len(words)

        # Count tokens
        if precomputed_tokens is not None:
            token_count = precomputed_tokens
        else:
            token_count = self._count_tokens(text)

        # Calculate rate
        fertility_rate = 0.0 if word_count == 0 else token_count / word_count

        return FertilityResult(
            text=text,
            word_count=word_count,
            token_count=token_count,
            fertility_rate=fertility_rate,
            metadata={
                "estimation_method": self._get_estimation_method(),
            },
        )

    def calculate_batch(
        self,
        texts: list[str],
        precomputed_tokens: list[int] | None = None,
    ) -> list[FertilityResult]:
        """Calculate fertility rate for multiple texts.

        Args:
            texts: List of texts to analyze.
            precomputed_tokens: Pre-computed token counts (optional).

        Returns:
            List of fertility results.
        """
        if precomputed_tokens is None:
            precomputed_tokens = [None] * len(texts)  # type: ignore[list-item]

        return [
            self.calculate(text, tokens)
            for text, tokens in zip(texts, precomputed_tokens, strict=True)
        ]

    def calculate_aggregate(
        self, texts: list[str]
    ) -> dict[str, float]:
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
        """Count tokens in text.

        Uses tokenizer if available, otherwise estimates.

        Args:
            text: Text to tokenize.

        Returns:
            Token count.
        """
        # Use custom counter if provided
        if self._token_counter is not None:
            return self._token_counter(text)

        # Use tokenizer if provided
        if self._tokenizer is not None:
            return len(self._tokenizer.encode(text))

        # Fall back to estimation
        return self._estimate_tokens(text)

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count using heuristics.

        Approximation based on:
        - Ukrainian typically tokenizes at ~1.3x words
        - Punctuation adds tokens
        - Numbers often split into multiple tokens

        Args:
            text: Text to estimate tokens for.

        Returns:
            Estimated token count.
        """
        words = WORD_PATTERN.findall(text)
        word_count = len(words)

        # Base estimation
        base_tokens = int(word_count * self.DEFAULT_TOKEN_FACTOR)

        # Add for punctuation
        punctuation_count = len(re.findall(r"[.,!?;:()\"«»—–-]", text))
        punct_tokens = punctuation_count // 2  # Not all punct is separate token

        # Add for numbers (often split)
        numbers = re.findall(r"\d+", text)
        number_tokens = sum(len(n) // 2 for n in numbers)  # Rough estimate

        return max(1, base_tokens + punct_tokens + number_tokens)

    def _get_estimation_method(self) -> str:
        """Get the method used for token counting."""
        if self._token_counter is not None:
            return "custom_counter"
        if self._tokenizer is not None:
            return "tokenizer"
        return "estimation"


def calculate_fertility(
    text: str,
    token_count: int | None = None,
) -> float:
    """Calculate fertility rate for text.

    Convenience function for simple usage.

    Args:
        text: Text to analyze.
        token_count: Pre-computed token count.

    Returns:
        Fertility rate (tokens / words).
    """
    calculator = FertilityCalculator()
    result = calculator.calculate(text, token_count)
    return result.fertility_rate


def evaluate_fertility_quality(fertility_rate: float) -> str:
    """Evaluate quality based on fertility rate.

    Args:
        fertility_rate: Calculated fertility rate.

    Returns:
        Quality level string.
    """
    if fertility_rate < 1.5:
        return "excellent"
    if fertility_rate < 2.0:
        return "good"
    if fertility_rate < 2.5:
        return "acceptable"
    if fertility_rate < 3.0:
        return "poor"
    return "very_poor"
