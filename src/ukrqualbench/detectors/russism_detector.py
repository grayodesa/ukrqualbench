"""Russism detector for Ukrainian text.

Detects Russian calques, syntactic patterns, and morphological errors
that indicate non-native Ukrainian language usage.

Categories:
- lexical: Word-level calques from Russian
- syntactic: Russian syntactic patterns in Ukrainian
- morphological: Russian morphological patterns
- phraseological: Calqued phrases from Russian
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ukrqualbench.detectors.base import (
    SEVERITY_WEIGHTS,
    BaseDetector,
    CompiledPattern,
    DetectionResult,
    DetectionSeverity,
)


class RussismDetector(BaseDetector):
    """Detector for russisms in Ukrainian text.

    Russisms are linguistic features borrowed from or influenced by Russian
    that are considered non-standard in Ukrainian. These include:

    - Lexical calques: Direct word translations
    - Syntactic patterns: Russian sentence structures
    - Morphological errors: Wrong word forms
    - Phraseological calques: Translated idioms

    Example:
        >>> detector = RussismDetector()
        >>> result = detector.detect("Прийняти участь у заході.")
        >>> result.count
        1
        >>> result.matches[0].correction
        'взяти участь'
    """

    def _get_default_dictionary_path(self) -> Path:
        """Return default path to russisms dictionary."""
        return Path(__file__).parent.parent.parent.parent / "data" / "dictionaries" / "russisms.json"

    def _load_patterns(self, data: dict[str, Any]) -> list[CompiledPattern]:
        """Load and compile russism patterns.

        Args:
            data: Parsed JSON dictionary data.

        Returns:
            List of compiled patterns.
        """
        patterns: list[CompiledPattern] = []
        categories = data.get("categories", {})

        for category_name, category_data in categories.items():
            if not isinstance(category_data, dict):
                continue

            category_patterns = category_data.get("patterns", [])
            for i, pattern_data in enumerate(category_patterns):
                pattern_str = pattern_data.get("pattern")
                if not pattern_str:
                    continue

                severity = pattern_data.get("severity", "medium")
                weight = SEVERITY_WEIGHTS.get(
                    DetectionSeverity(severity), 1.0
                )

                compiled = self._compile_pattern(
                    pattern_str=pattern_str,
                    pattern_id=f"russism_{category_name}_{i:03d}",
                    category=category_name,
                    severity_str=severity,
                    correction=pattern_data.get("correction"),
                    description=category_data.get("description"),
                    weight=weight,
                )

                if compiled:
                    patterns.append(compiled)

        return patterns

    def detect_with_context(
        self, text: str, token_count: int | None = None, context_chars: int = 50
    ) -> DetectionResult:
        """Detect russisms with surrounding context.

        Args:
            text: Text to analyze.
            token_count: Pre-computed token count.
            context_chars: Characters of context to include.

        Returns:
            Detection result with context in metadata.
        """
        result = self.detect(text, token_count)

        # Add context for each match
        contexts: list[dict[str, str]] = []
        for match in result.matches:
            start = max(0, match.start - context_chars)
            end = min(len(text), match.end + context_chars)
            context = text[start:end]

            # Mark the matched portion
            rel_start = match.start - start
            rel_end = match.end - start
            marked = (
                context[:rel_start]
                + ">>>"
                + context[rel_start:rel_end]
                + "<<<"
                + context[rel_end:]
            )
            contexts.append({
                "pattern_id": match.pattern_id,
                "context": marked,
            })

        result.metadata["contexts"] = contexts
        return result

    def get_severity_breakdown(
        self, result: DetectionResult
    ) -> dict[str, int]:
        """Get breakdown of matches by severity.

        Args:
            result: Detection result.

        Returns:
            Dictionary with counts per severity level.
        """
        breakdown: dict[str, int] = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
        }

        for match in result.matches:
            breakdown[match.severity.value] += 1

        return breakdown

    def calculate_quality_score(
        self, result: DetectionResult
    ) -> float:
        """Calculate quality score based on russism density.

        Score ranges from 0 to 100, where:
        - 100: No russisms detected
        - 0: Very high russism density

        Args:
            result: Detection result.

        Returns:
            Quality score (0-100).
        """
        if result.total_tokens == 0:
            return 100.0

        # Use weighted rate for more nuanced scoring
        weighted_rate = result.weighted_rate_per_1k

        # Thresholds (russisms per 1K tokens)
        # Gold: < 1.0, Silver: < 3.0, Bronze: < 5.0, Caution: < 10.0
        if weighted_rate < 1.0:
            return 100.0
        if weighted_rate < 3.0:
            return 90.0 - (weighted_rate - 1.0) * 5
        if weighted_rate < 5.0:
            return 80.0 - (weighted_rate - 3.0) * 5
        if weighted_rate < 10.0:
            return 70.0 - (weighted_rate - 5.0) * 4

        # Below 10 is poor quality
        return max(0.0, 50.0 - (weighted_rate - 10.0) * 2)

    def get_corrections(
        self, result: DetectionResult
    ) -> list[dict[str, str]]:
        """Get list of corrections for detected russisms.

        Args:
            result: Detection result.

        Returns:
            List of correction suggestions.
        """
        corrections: list[dict[str, str]] = []

        for match in result.matches:
            if match.correction:
                corrections.append({
                    "original": match.matched_text,
                    "correction": match.correction,
                    "category": match.category,
                    "position": f"{match.start}-{match.end}",
                })

        return corrections


def create_russism_detector(
    dictionary_path: Path | None = None,
) -> RussismDetector:
    """Factory function to create and initialize a russism detector.

    Args:
        dictionary_path: Optional custom dictionary path.

    Returns:
        Initialized russism detector.
    """
    detector = RussismDetector(dictionary_path)
    detector.initialize()
    return detector
