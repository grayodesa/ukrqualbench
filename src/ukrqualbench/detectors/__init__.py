"""Automatic detectors for language quality metrics.

Detectors for:
- Russisms (lexical, syntactic, prepositional)
- Anglicisms (calques)
- Positive markers (vocative, particles, diminutives)
- Fertility rate (tokens/word ratio)
"""

from ukrqualbench.detectors.anglicism_detector import (
    AnglicismDetector,
    create_anglicism_detector,
)
from ukrqualbench.detectors.base import (
    SEVERITY_WEIGHTS,
    BaseDetector,
    CompiledPattern,
    DetectionMatch,
    DetectionResult,
    DetectionSeverity,
)
from ukrqualbench.detectors.calque_detector import (
    CalqueDetectionConfig,
    JudgeBasedCalqueDetector,
    create_calque_detector,
)
from ukrqualbench.detectors.fertility import (
    FertilityCalculator,
    FertilityResult,
    calculate_fertility,
    evaluate_fertility_quality,
)
from ukrqualbench.detectors.positive_marker_detector import (
    PositiveMarkerDetector,
    create_positive_marker_detector,
)
from ukrqualbench.detectors.russism_detector import (
    RussismDetector,
    create_russism_detector,
)

__all__ = [
    "SEVERITY_WEIGHTS",
    "AnglicismDetector",
    "BaseDetector",
    "CalqueDetectionConfig",
    "CompiledPattern",
    "DetectionMatch",
    "DetectionResult",
    "DetectionSeverity",
    "FertilityCalculator",
    "FertilityResult",
    "JudgeBasedCalqueDetector",
    "PositiveMarkerDetector",
    "RussismDetector",
    "calculate_fertility",
    "create_anglicism_detector",
    "create_calque_detector",
    "create_positive_marker_detector",
    "create_russism_detector",
    "evaluate_fertility_quality",
]
