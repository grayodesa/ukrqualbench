"""Positive marker detector for Ukrainian text.

Detects positive markers of native Ukrainian language usage.
These markers indicate authentic, native-sounding Ukrainian:

Categories:
- vocative: Vocative case usage (кличний відмінок)
- particles: Ukrainian particles adding expressiveness
- conjunctions: Native Ukrainian conjunctions
- diminutives: Diminutive and affectionate forms
- impersonal: Ukrainian impersonal constructions
- phraseology: Native Ukrainian idioms and expressions
- interjections: Ukrainian interjections and exclamations

Higher positive marker rates indicate more natural, native Ukrainian.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ukrqualbench.detectors.base import (
    BaseDetector,
    CompiledPattern,
    DetectionResult,
)

# Default weights for positive marker categories
DEFAULT_CATEGORY_WEIGHTS: dict[str, float] = {
    "vocative": 2.0,  # Most distinctive Ukrainian feature
    "particles": 1.0,
    "conjunctions": 1.0,
    "diminutives": 1.5,
    "impersonal": 1.0,
    "phraseology": 1.5,
    "interjections": 0.5,  # Less formal, context-dependent
}


class PositiveMarkerDetector(BaseDetector):
    """Detector for positive markers of native Ukrainian.

    Unlike error detectors, this finds GOOD features that indicate
    authentic Ukrainian language usage. Higher rates are better.

    Positive markers include:
    - Vocative case: "Пане Андрію" instead of "Пан Андрій"
    - Particles: же, бо, адже, хіба, невже
    - Native conjunctions: проте, однак, утім, зате, отже
    - Diminutives: -очк-, -еньк-, -усіньк- suffixes
    - Impersonal constructions: "треба", "варто", "годі"
    - Native idioms: "як кіт наплакав", "ні пуху ні пера"

    Example:
        >>> detector = PositiveMarkerDetector()
        >>> result = detector.detect("Пане Андрію, як справи?")
        >>> result.count
        1
        >>> result.matches[0].category
        'vocative'
    """

    def _get_default_dictionary_path(self) -> Path:
        """Return default path to positive markers dictionary."""
        return (
            Path(__file__).parent.parent.parent.parent
            / "data"
            / "dictionaries"
            / "positive_markers.json"
        )

    def _load_patterns(self, data: dict[str, Any]) -> list[CompiledPattern]:
        """Load and compile positive marker patterns.

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

            # Get category weight
            category_weight = category_data.get(
                "weight", DEFAULT_CATEGORY_WEIGHTS.get(category_name, 1.0)
            )

            category_patterns = category_data.get("patterns", [])
            for i, pattern_data in enumerate(category_patterns):
                pattern_str = pattern_data.get("pattern")
                if not pattern_str:
                    continue

                description = pattern_data.get("description") or pattern_data.get("example")

                compiled = self._compile_pattern(
                    pattern_str=pattern_str,
                    pattern_id=f"positive_{category_name}_{i:03d}",
                    category=category_name,
                    # All positive markers have "low" severity (they're good!)
                    severity_str="low",
                    correction=None,  # No correction needed
                    description=description,
                    weight=category_weight,
                )

                if compiled:
                    patterns.append(compiled)

        return patterns

    def detect(self, text: str, token_count: int | None = None) -> DetectionResult:
        """Detect positive markers in text.

        Args:
            text: Text to analyze.
            token_count: Pre-computed token count.

        Returns:
            Detection result with positive markers found.
        """
        # Call parent detect
        result = super().detect(text, token_count)

        # Override severity to show these are positive (not errors)
        # We keep severity as LOW but interpret it differently
        result.metadata["is_positive"] = True
        result.metadata["interpretation"] = "Higher counts indicate better quality"

        return result

    def calculate_nativeness_score(self, result: DetectionResult) -> float:
        """Calculate nativeness score based on positive marker density.

        Higher weighted rate = more native-sounding text.

        Score thresholds (per 1K tokens):
        - Excellent (15+): Very natural, native Ukrainian
        - Good (10-15): Good native markers present
        - Acceptable (5-10): Some native markers
        - Poor (2-5): Few native markers
        - Very poor (<2): Almost no native markers

        Args:
            result: Detection result.

        Returns:
            Nativeness score (0-100).
        """
        if result.total_tokens == 0:
            return 0.0

        weighted_rate = result.weighted_rate_per_1k

        # Thresholds based on positive_markers.json scoring section
        if weighted_rate >= 15.0:
            return 100.0  # Excellent
        if weighted_rate >= 10.0:
            return 85.0 + (weighted_rate - 10.0) * 3  # Good
        if weighted_rate >= 5.0:
            return 60.0 + (weighted_rate - 5.0) * 5  # Acceptable
        if weighted_rate >= 2.0:
            return 30.0 + (weighted_rate - 2.0) * 10  # Poor

        # Very poor
        return weighted_rate * 15  # 0-30

    def get_category_breakdown(self, result: DetectionResult) -> dict[str, dict[str, Any]]:
        """Get breakdown of positive markers by category.

        Args:
            result: Detection result.

        Returns:
            Dictionary with counts and weights per category.
        """
        breakdown: dict[str, dict[str, Any]] = {}

        for match in result.matches:
            if match.category not in breakdown:
                breakdown[match.category] = {
                    "count": 0,
                    "total_weight": 0.0,
                    "examples": [],
                }
            breakdown[match.category]["count"] += 1
            breakdown[match.category]["total_weight"] += match.weight

            # Store first few examples
            examples = breakdown[match.category]["examples"]
            if isinstance(examples, list) and len(examples) < 3:
                examples.append(match.matched_text)

        return breakdown

    def get_missing_categories(self, result: DetectionResult) -> list[str]:
        """Get categories with no detected markers.

        Useful for identifying areas to improve Ukrainian nativeness.

        Args:
            result: Detection result.

        Returns:
            List of missing category names.
        """
        # All possible categories
        all_categories = set(DEFAULT_CATEGORY_WEIGHTS.keys())

        # Categories found in result
        found_categories = {m.category for m in result.matches}

        return list(all_categories - found_categories)

    def get_improvement_suggestions(self, result: DetectionResult) -> list[dict[str, str]]:
        """Get suggestions for improving Ukrainian nativeness.

        Args:
            result: Detection result.

        Returns:
            List of improvement suggestions.
        """
        suggestions: list[dict[str, str]] = []
        missing = self.get_missing_categories(result)

        category_suggestions = {
            "vocative": "Use vocative case for addresses: 'Пане Іване' instead of 'Пан Іван'",
            "particles": "Add emphatic particles: же, бо, адже, хіба, невже",
            "conjunctions": "Use native conjunctions: проте, однак, утім, зате, отже",
            "diminutives": "Include diminutives where appropriate: -очк-, -еньк-",
            "impersonal": "Use impersonal constructions: треба, варто, слід, годі",
            "phraseology": "Include Ukrainian idioms and set expressions",
            "interjections": "Add natural interjections: ой, ех, агов, ану",
        }

        for category in missing:
            if category in category_suggestions:
                suggestions.append(
                    {
                        "category": category,
                        "suggestion": category_suggestions[category],
                        "priority": "high" if category in ("vocative", "particles") else "medium",
                    }
                )

        # Sort by priority (high first)
        suggestions.sort(key=lambda x: 0 if x["priority"] == "high" else 1)

        return suggestions

    def analyze_balance(self, result: DetectionResult) -> dict[str, Any]:
        """Analyze the balance of positive markers across categories.

        Args:
            result: Detection result.

        Returns:
            Analysis with balance metrics.
        """
        breakdown = self.get_category_breakdown(result)

        total_markers = result.count
        category_percentages: dict[str, float] = {}

        for category, stats in breakdown.items():
            if total_markers > 0:
                category_percentages[category] = (stats["count"] / total_markers) * 100
            else:
                category_percentages[category] = 0.0

        # Calculate diversity score (more diverse = better)
        categories_present = len(breakdown)
        max_categories = len(DEFAULT_CATEGORY_WEIGHTS)
        diversity_score = (categories_present / max_categories) * 100 if max_categories > 0 else 0

        return {
            "total_markers": total_markers,
            "categories_present": categories_present,
            "max_categories": max_categories,
            "diversity_score": diversity_score,
            "category_percentages": category_percentages,
            "missing_categories": self.get_missing_categories(result),
        }


def create_positive_marker_detector(
    dictionary_path: Path | None = None,
) -> PositiveMarkerDetector:
    """Factory function to create and initialize a positive marker detector.

    Args:
        dictionary_path: Optional custom dictionary path.

    Returns:
        Initialized positive marker detector.
    """
    detector = PositiveMarkerDetector(dictionary_path)
    detector.initialize()
    return detector
