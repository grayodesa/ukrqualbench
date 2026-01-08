"""Anglicism detector for Ukrainian text.

Detects English calques and borrowings in Ukrainian text.
Anglicisms are treated more leniently than russisms, as they are
often acceptable in technical and business contexts.

Categories:
- lexical: English word borrowings and calques
- syntactic: English syntactic patterns in Ukrainian
- it_jargon: IT-related anglicisms (more acceptable)
- business_jargon: Business-related anglicisms
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from ukrqualbench.detectors.base import (
    SEVERITY_WEIGHTS,
    BaseDetector,
    CompiledPattern,
    DetectionMatch,
    DetectionResult,
    DetectionSeverity,
)


class AnglicismDetector(BaseDetector):
    """Detector for anglicisms in Ukrainian text.

    Anglicisms are linguistic features borrowed from or influenced by English.
    They are generally less problematic than russisms in Ukrainian,
    especially in technical and business contexts.

    The detector applies context-aware scoring:
    - IT jargon has reduced penalty in technical contexts
    - Business jargon is more acceptable in formal business writing
    - General lexical anglicisms are penalized normally

    Example:
        >>> detector = AnglicismDetector()
        >>> result = detector.detect("Чекаю на ваш фідбек.")
        >>> result.count
        1
        >>> result.matches[0].correction
        'відгук / зворотний зв'язок'
    """

    # Default weight vs russisms (anglicisms are less critical)
    ANGLICISM_WEIGHT: ClassVar[float] = 0.5

    # Categories with reduced penalties in certain contexts
    TECHNICAL_CATEGORIES: ClassVar[set[str]] = {"it_jargon"}
    BUSINESS_CATEGORIES: ClassVar[set[str]] = {"business_jargon"}

    def _get_default_dictionary_path(self) -> Path:
        """Return default path to anglicisms dictionary."""
        return Path(__file__).parent.parent.parent.parent / "data" / "dictionaries" / "anglicisms.json"

    def _load_patterns(self, data: dict[str, Any]) -> list[CompiledPattern]:
        """Load and compile anglicism patterns.

        Args:
            data: Parsed JSON dictionary data.

        Returns:
            List of compiled patterns.
        """
        patterns: list[CompiledPattern] = []
        categories = data.get("categories", {})

        # Get global scoring settings
        scoring = data.get("scoring", {})
        weight_vs_russisms = scoring.get("weight_vs_russisms", self.ANGLICISM_WEIGHT)

        for category_name, category_data in categories.items():
            if not isinstance(category_data, dict):
                continue

            category_patterns = category_data.get("patterns", [])
            for i, pattern_data in enumerate(category_patterns):
                pattern_str = pattern_data.get("pattern")
                if not pattern_str:
                    continue

                severity = pattern_data.get("severity", "low")
                base_weight = SEVERITY_WEIGHTS.get(
                    DetectionSeverity(severity), 1.0
                )

                # Apply anglicism weight reduction
                weight = base_weight * weight_vs_russisms

                # Further reduce weight for technical/business categories
                if category_name in self.TECHNICAL_CATEGORIES:
                    weight *= 0.5
                elif category_name in self.BUSINESS_CATEGORIES:
                    weight *= 0.7

                compiled = self._compile_pattern(
                    pattern_str=pattern_str,
                    pattern_id=f"anglicism_{category_name}_{i:03d}",
                    category=category_name,
                    severity_str=severity,
                    correction=pattern_data.get("correction"),
                    description=pattern_data.get("note") or category_data.get("description"),
                    weight=weight,
                )

                if compiled:
                    patterns.append(compiled)

        return patterns

    def detect_with_context_awareness(
        self,
        text: str,
        is_technical: bool = False,
        is_business: bool = False,
        token_count: int | None = None,
    ) -> DetectionResult:
        """Detect anglicisms with context-aware scoring.

        Args:
            text: Text to analyze.
            is_technical: Whether text is technical/IT context.
            is_business: Whether text is business context.
            token_count: Pre-computed token count.

        Returns:
            Detection result with adjusted weights.
        """
        result = self.detect(text, token_count)

        # Adjust weights based on context
        if is_technical or is_business:
            adjusted_matches: list[DetectionMatch] = []
            for match in result.matches:
                new_match = match
                if is_technical and match.category in self.TECHNICAL_CATEGORIES:
                    # Further reduce weight for IT jargon in technical context
                    new_match = DetectionMatch(
                        start=match.start,
                        end=match.end,
                        matched_text=match.matched_text,
                        pattern_id=match.pattern_id,
                        category=match.category,
                        severity=match.severity,
                        correction=match.correction,
                        description=match.description,
                        weight=match.weight * 0.3,  # 70% reduction
                    )
                elif is_business and match.category in self.BUSINESS_CATEGORIES:
                    # Reduce weight for business jargon in business context
                    new_match = DetectionMatch(
                        start=match.start,
                        end=match.end,
                        matched_text=match.matched_text,
                        pattern_id=match.pattern_id,
                        category=match.category,
                        severity=match.severity,
                        correction=match.correction,
                        description=match.description,
                        weight=match.weight * 0.5,  # 50% reduction
                    )
                adjusted_matches.append(new_match)
            result.matches = adjusted_matches

        result.metadata["is_technical"] = is_technical
        result.metadata["is_business"] = is_business

        return result

    def get_category_breakdown(
        self, result: DetectionResult
    ) -> dict[str, dict[str, int | float]]:
        """Get breakdown of matches by category with statistics.

        Args:
            result: Detection result.

        Returns:
            Dictionary with counts and weights per category.
        """
        breakdown: dict[str, dict[str, int | float]] = {}

        for match in result.matches:
            if match.category not in breakdown:
                breakdown[match.category] = {
                    "count": 0,
                    "total_weight": 0.0,
                }
            breakdown[match.category]["count"] += 1
            breakdown[match.category]["total_weight"] += match.weight

        return breakdown

    def calculate_quality_score(
        self, result: DetectionResult, base_score: float = 100.0
    ) -> float:
        """Calculate quality score based on anglicism density.

        Anglicisms have less impact than russisms:
        - Light penalty for low/medium severity
        - Moderate penalty for high severity
        - Context affects final score

        Args:
            result: Detection result.
            base_score: Starting score before deductions.

        Returns:
            Quality score (0-100).
        """
        if result.total_tokens == 0:
            return base_score

        # Use weighted rate (already includes anglicism weight reduction)
        weighted_rate = result.weighted_rate_per_1k

        # Anglicisms are less critical, so thresholds are higher
        # Acceptable: < 5.0 per 1K
        # Warning: < 10.0 per 1K
        # Excessive: > 15.0 per 1K

        if weighted_rate < 2.0:
            return base_score  # Minimal impact
        if weighted_rate < 5.0:
            return base_score - (weighted_rate - 2.0) * 2  # Light penalty
        if weighted_rate < 10.0:
            return base_score - 6 - (weighted_rate - 5.0) * 3  # Moderate penalty
        if weighted_rate < 15.0:
            return base_score - 21 - (weighted_rate - 10.0) * 4

        # Excessive anglicisms
        return max(0.0, base_score - 41 - (weighted_rate - 15.0) * 2)

    def get_acceptable_in_context(
        self, result: DetectionResult, context: str = "general"
    ) -> list[DetectionMatch]:
        """Filter matches that are acceptable in given context.

        Args:
            result: Detection result.
            context: Context type ("general", "technical", "business").

        Returns:
            List of matches acceptable in context.
        """
        acceptable: list[DetectionMatch] = []

        for match in result.matches:
            if (context == "technical" and match.category in self.TECHNICAL_CATEGORIES) or (context == "business" and match.category in self.BUSINESS_CATEGORIES):
                acceptable.append(match)

        return acceptable

    def get_corrections(
        self, result: DetectionResult
    ) -> list[dict[str, str]]:
        """Get list of corrections for detected anglicisms.

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
                    "context_note": self._get_context_note(match.category),
                })

        return corrections

    def _get_context_note(self, category: str) -> str:
        """Get context acceptability note for category."""
        if category in self.TECHNICAL_CATEGORIES:
            return "Acceptable in technical writing"
        if category in self.BUSINESS_CATEGORIES:
            return "Common in business context"
        return "Consider Ukrainian alternative"


def create_anglicism_detector(
    dictionary_path: Path | None = None,
) -> AnglicismDetector:
    """Factory function to create and initialize an anglicism detector.

    Args:
        dictionary_path: Optional custom dictionary path.

    Returns:
        Initialized anglicism detector.
    """
    detector = AnglicismDetector(dictionary_path)
    detector.initialize()
    return detector
