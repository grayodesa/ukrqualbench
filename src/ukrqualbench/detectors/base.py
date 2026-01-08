"""Base classes for language quality detectors.

Provides common infrastructure for detecting:
- Russisms (lexical, syntactic, morphological)
- Anglicisms (calques, IT jargon)
- Positive markers (vocative, particles, idioms)
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator


class DetectionSeverity(str, Enum):
    """Severity levels for detected issues."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class DetectionMatch:
    """Single detection match in text."""

    start: int
    end: int
    matched_text: str
    pattern_id: str
    category: str
    severity: DetectionSeverity = DetectionSeverity.MEDIUM
    correction: str | None = None
    description: str | None = None
    weight: float = 1.0

    @property
    def span(self) -> tuple[int, int]:
        """Return match span as tuple."""
        return (self.start, self.end)


@dataclass
class DetectionResult:
    """Complete detection results for a text."""

    text: str
    matches: list[DetectionMatch] = field(default_factory=list)
    total_tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def count(self) -> int:
        """Total number of matches."""
        return len(self.matches)

    @property
    def rate_per_1k(self) -> float:
        """Detection rate per 1000 tokens."""
        if self.total_tokens == 0:
            return 0.0
        return (self.count / self.total_tokens) * 1000

    @property
    def weighted_rate_per_1k(self) -> float:
        """Weighted detection rate per 1000 tokens."""
        if self.total_tokens == 0:
            return 0.0
        total_weight = sum(m.weight for m in self.matches)
        return (total_weight / self.total_tokens) * 1000

    def by_category(self) -> dict[str, list[DetectionMatch]]:
        """Group matches by category."""
        result: dict[str, list[DetectionMatch]] = {}
        for match in self.matches:
            if match.category not in result:
                result[match.category] = []
            result[match.category].append(match)
        return result

    def by_severity(self) -> dict[DetectionSeverity, list[DetectionMatch]]:
        """Group matches by severity."""
        result: dict[DetectionSeverity, list[DetectionMatch]] = {}
        for match in self.matches:
            if match.severity not in result:
                result[match.severity] = []
            result[match.severity].append(match)
        return result


@dataclass
class CompiledPattern:
    """Pre-compiled regex pattern with metadata."""

    pattern_id: str
    regex: re.Pattern[str]
    category: str
    severity: DetectionSeverity
    correction: str | None = None
    description: str | None = None
    weight: float = 1.0


class BaseDetector(ABC):
    """Abstract base class for language detectors."""

    def __init__(self, dictionary_path: Path | None = None) -> None:
        """Initialize detector.

        Args:
            dictionary_path: Path to dictionary JSON file.
        """
        self.dictionary_path = dictionary_path
        self._patterns: list[CompiledPattern] = []
        self._initialized = False

    @abstractmethod
    def _get_default_dictionary_path(self) -> Path:
        """Return default path to dictionary file."""
        ...

    @abstractmethod
    def _load_patterns(self, data: dict[str, Any]) -> list[CompiledPattern]:
        """Load and compile patterns from dictionary data.

        Args:
            data: Parsed JSON dictionary data.

        Returns:
            List of compiled patterns.
        """
        ...

    def initialize(self) -> None:
        """Load dictionary and compile patterns."""
        if self._initialized:
            return

        path = self.dictionary_path or self._get_default_dictionary_path()

        if not path.exists():
            raise FileNotFoundError(f"Dictionary not found: {path}")

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        self._patterns = self._load_patterns(data)
        self._initialized = True

    def detect(self, text: str, token_count: int | None = None) -> DetectionResult:
        """Detect patterns in text.

        Args:
            text: Text to analyze.
            token_count: Pre-computed token count (optional).

        Returns:
            Detection result with all matches.
        """
        if not self._initialized:
            self.initialize()

        matches: list[DetectionMatch] = []

        for pattern in self._patterns:
            for match in pattern.regex.finditer(text):
                detection = DetectionMatch(
                    start=match.start(),
                    end=match.end(),
                    matched_text=match.group(),
                    pattern_id=pattern.pattern_id,
                    category=pattern.category,
                    severity=pattern.severity,
                    correction=pattern.correction,
                    description=pattern.description,
                    weight=pattern.weight,
                )
                matches.append(detection)

        # Remove overlapping matches (keep first/longest)
        matches = self._remove_overlaps(matches)

        # Estimate tokens if not provided
        if token_count is None:
            token_count = self._estimate_tokens(text)

        return DetectionResult(
            text=text,
            matches=matches,
            total_tokens=token_count,
        )

    def detect_batch(
        self, texts: list[str], token_counts: list[int] | None = None
    ) -> list[DetectionResult]:
        """Detect patterns in multiple texts.

        Args:
            texts: List of texts to analyze.
            token_counts: Pre-computed token counts (optional).

        Returns:
            List of detection results.
        """
        if token_counts is None:
            token_counts = [None] * len(texts)  # type: ignore[list-item]

        return [
            self.detect(text, count) for text, count in zip(texts, token_counts, strict=True)
        ]

    def iter_matches(self, text: str) -> Iterator[DetectionMatch]:
        """Iterate over matches in text.

        Args:
            text: Text to analyze.

        Yields:
            Detection matches.
        """
        result = self.detect(text)
        yield from result.matches

    def _remove_overlaps(
        self, matches: list[DetectionMatch]
    ) -> list[DetectionMatch]:
        """Remove overlapping matches, keeping the first/longest."""
        if not matches:
            return []

        # Sort by start position, then by length (descending)
        sorted_matches = sorted(
            matches, key=lambda m: (m.start, -(m.end - m.start))
        )

        result: list[DetectionMatch] = []
        last_end = -1

        for match in sorted_matches:
            if match.start >= last_end:
                result.append(match)
                last_end = match.end

        return result

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.

        Uses simple heuristic: ~1.3 tokens per word for Ukrainian.

        Args:
            text: Text to analyze.

        Returns:
            Estimated token count.
        """
        words = len(text.split())
        return max(1, int(words * 1.3))

    def _compile_pattern(
        self,
        pattern_str: str,
        pattern_id: str,
        category: str,
        severity_str: str,
        correction: str | None = None,
        description: str | None = None,
        weight: float = 1.0,
    ) -> CompiledPattern | None:
        """Compile a single pattern.

        Args:
            pattern_str: Regex pattern string.
            pattern_id: Unique pattern identifier.
            category: Pattern category.
            severity_str: Severity level string.
            correction: Suggested correction.
            description: Pattern description.
            weight: Pattern weight for scoring.

        Returns:
            Compiled pattern or None if invalid.
        """
        try:
            regex = re.compile(pattern_str, re.IGNORECASE | re.UNICODE)
            severity = DetectionSeverity(severity_str)
            return CompiledPattern(
                pattern_id=pattern_id,
                regex=regex,
                category=category,
                severity=severity,
                correction=correction,
                description=description,
                weight=weight,
            )
        except (re.error, ValueError):
            return None

    @property
    def pattern_count(self) -> int:
        """Number of loaded patterns."""
        return len(self._patterns)

    def get_statistics(self) -> dict[str, Any]:
        """Get detector statistics.

        Returns:
            Dictionary with pattern counts by category.
        """
        if not self._initialized:
            self.initialize()

        stats: dict[str, Any] = {
            "total_patterns": len(self._patterns),
            "by_category": {},
            "by_severity": {},
        }

        for pattern in self._patterns:
            # Count by category
            if pattern.category not in stats["by_category"]:
                stats["by_category"][pattern.category] = 0
            stats["by_category"][pattern.category] += 1

            # Count by severity
            severity_key = pattern.severity.value
            if severity_key not in stats["by_severity"]:
                stats["by_severity"][severity_key] = 0
            stats["by_severity"][severity_key] += 1

        return stats


# Severity weights for scoring
SEVERITY_WEIGHTS: dict[DetectionSeverity, float] = {
    DetectionSeverity.CRITICAL: 3.0,
    DetectionSeverity.HIGH: 2.0,
    DetectionSeverity.MEDIUM: 1.0,
    DetectionSeverity.LOW: 0.5,
}
