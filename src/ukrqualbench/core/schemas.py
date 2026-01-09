"""Pydantic schemas for UkrQualBench data structures.

These schemas provide validated, type-safe data structures for:
- Dataset tasks (Block A, B)
- Model responses and judge verdicts
- Evaluation results and scores
- Comparison records
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# ============================================================================
# Enums
# ============================================================================


class ErrorSeverity(str, Enum):
    """Severity levels for detected language errors."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TaskDifficulty(str, Enum):
    """Task difficulty levels."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class WinnerChoice(str, Enum):
    """Possible winners in pairwise comparison."""

    A = "A"
    B = "B"
    TIE = "tie"


class ConfidenceLevel(str, Enum):
    """Judge confidence levels."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class PositionOrder(str, Enum):
    """Order in which responses were presented."""

    AB = "AB"
    BA = "BA"


class Badge(str, Enum):
    """Quality badges based on ELO and russism rate."""

    GOLD = "gold"
    SILVER = "silver"
    BRONZE = "bronze"
    CAUTION = "caution"
    NOT_RECOMMENDED = "not_recommended"
    NONE = "none"


# ============================================================================
# Block A: Tasks with Reference Answers
# ============================================================================


class MultipleChoiceTask(BaseModel):
    """Multiple choice task (A1: ZNO, error detection)."""

    id: str = Field(..., description="Unique task identifier")
    type: Literal["multiple_choice"] = "multiple_choice"
    category: str = Field(..., description="Category (orthography, punctuation, etc.)")
    subcategory: str | None = Field(None, description="Optional subcategory")
    prompt: str = Field(..., description="The question text")
    options: list[str] = Field(..., min_length=2, max_length=10)
    correct: str = Field(..., description="Correct answer letter (A, B, C, D)")
    explanation: str | None = Field(None, description="Explanation of correct answer")
    difficulty: TaskDifficulty = TaskDifficulty.MEDIUM
    source: str = Field(..., description="Data source (ZNO-2023, UA-GEC, etc.)")

    @field_validator("correct")
    @classmethod
    def validate_correct(cls, v: str, info: object) -> str:
        """Ensure correct answer is a valid option letter."""
        if not v.isalpha() or len(v) != 1:
            raise ValueError("Correct answer must be a single letter")
        return v.upper()


class GECTask(BaseModel):
    """Grammar Error Correction task (A2: from UA-GEC)."""

    id: str
    type: Literal["gec"] = "gec"
    category: str = Field(..., description="Error category (russism, grammar_case, etc.)")
    input: str = Field(..., description="Text with errors")
    expected_output: str = Field(..., description="Corrected text")
    errors: list[ErrorAnnotation] = Field(default_factory=list, description="Annotated errors")
    source: str = "UA-GEC"
    difficulty: TaskDifficulty = TaskDifficulty.MEDIUM


class ErrorAnnotation(BaseModel):
    """Single error annotation in GEC task."""

    start: int = Field(..., ge=0, description="Start position in input")
    end: int = Field(..., ge=0, description="End position in input")
    error_type: str = Field(..., description="Type of error")
    correction: str = Field(..., description="Suggested correction")
    severity: ErrorSeverity = ErrorSeverity.MEDIUM


class TranslationTask(BaseModel):
    """Translation task (A3: EN→UK and RU→UK)."""

    id: str
    type: Literal["translation"] = "translation"
    source_lang: Literal["en", "ru"] = Field(..., description="Source language")
    target_lang: Literal["uk"] = "uk"
    source: str = Field(..., description="Source text to translate")
    reference: str = Field(..., description="Reference translation")
    traps: list[str] = Field(
        default_factory=list,
        description="Translation traps (false friends, calques)",
    )
    trap_type: str | None = Field(None, description="Type of translation trap")


class FalsePositiveTask(BaseModel):
    """False positive test (A4: correct texts that shouldn't be flagged)."""

    id: str
    type: Literal["false_positive"] = "false_positive"
    text: str = Field(..., description="Text from classic literature")
    author: str | None = Field(None, description="Author name")
    is_correct: bool = Field(True, description="Whether text is correct")
    notes: str | None = Field(None, description="Notes about potential false flags")
    acceptable_flags: list[str] = Field(
        default_factory=list,
        description="Error types that are acceptable to flag",
    )


class PositiveMarkerTask(BaseModel):
    """Positive marker test (A5: native language markers)."""

    id: str
    type: Literal["positive_marker"] = "positive_marker"
    category: str = Field(..., description="Marker category (vocative, particles, etc.)")
    context: str = Field(..., description="Context for the marker")
    native_form: str = Field(..., description="Native Ukrainian form")
    non_native_forms: list[str] = Field(..., description="Non-native alternatives")
    marker_regex: str = Field(..., description="Regex pattern for detection")


# ============================================================================
# Block B: Generation Tasks (Pairwise Evaluation)
# ============================================================================


class FreeGenerationTask(BaseModel):
    """Free generation task (B1)."""

    id: str
    type: Literal["free_generation"] = "free_generation"
    category: str = Field(..., description="Category (explanation, advice, creative, technical)")
    prompt: str = Field(..., description="Generation prompt")
    min_tokens: int = Field(50, ge=10, description="Minimum response tokens")
    max_tokens: int = Field(300, ge=50, description="Maximum response tokens")


class AdversarialTask(BaseModel):
    """Adversarial task (B2: test resistance to bad Ukrainian)."""

    id: str
    type: Literal["adversarial"] = "adversarial"
    category: str = Field(..., description="Adversarial category (russism_trap, anglicism_trap)")
    prompt: str = Field(..., description="Prompt containing traps")
    traps_in_prompt: list[str] = Field(..., description="List of traps/errors in the prompt")
    instruction: str = Field(..., description="Instruction that implies correct Ukrainian")


class LongContextTask(BaseModel):
    """Long context task (B3: test language degradation)."""

    id: str
    type: Literal["long_context"] = "long_context"
    category: str = Field(..., description="Task category (consistency, degradation)")
    messages: list[dict[str, str]] = Field(..., description="Conversation messages")
    total_tokens: int = Field(..., ge=1000, description="Approximate token count")
    checkpoints: list[float] = Field(..., description="Token percentages to check quality")
    metrics: list[str] = Field(..., description="Metrics to track at checkpoints")


# ============================================================================
# Model Response and Judge Verdict
# ============================================================================


class ModelResponseData(BaseModel):
    """Model response data structure."""

    text: str = Field(..., description="Generated text")
    tokens_used: int = Field(..., ge=0, description="Tokens consumed")
    latency_ms: float = Field(..., ge=0, description="Response latency in ms")
    model_id: str = Field(..., description="Model identifier")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    cost_usd: float = Field(0.0, ge=0, description="Cost in USD")


class JudgeVerdictData(BaseModel):
    """Judge verdict data structure."""

    winner: WinnerChoice = Field(..., description="Which response won")
    confidence: ConfidenceLevel = Field(..., description="Judge confidence")
    reasoning: str = Field(..., description="Brief explanation")
    raw_response: str = Field("", description="Raw JSON from judge")
    latency_ms: float = Field(..., ge=0, description="Judge latency in ms")


class ComparisonRecordData(BaseModel):
    """Complete comparison record for audit trail."""

    comparison_id: str = Field(..., description="Idempotent key")
    prompt_id: str = Field(..., description="Prompt identifier")
    model_a_id: str = Field(..., description="First model ID")
    model_b_id: str = Field(..., description="Second model ID")
    response_a: ModelResponseData = Field(..., description="Response from A")
    response_b: ModelResponseData = Field(..., description="Response from B")
    verdict: JudgeVerdictData = Field(..., description="Judge verdict")
    judge_id: str = Field(..., description="Judge model ID")
    timestamp: datetime = Field(default_factory=datetime.now, description="Comparison timestamp")
    position_order: PositionOrder = Field(..., description="Presentation order for bias tracking")


# ============================================================================
# Scores and Results
# ============================================================================


class BlockAScores(BaseModel):
    """Block A (calibration tests) scores."""

    mc_accuracy: float = Field(..., ge=0, le=1, description="MC accuracy")
    gec_f1: float = Field(..., ge=0, le=1, description="GEC F1 score")
    translation_comet: float = Field(..., ge=0, le=1, description="COMET score")
    false_positive_rate: float = Field(..., ge=0, le=1, description="FP rate (lower=better)")
    positive_markers_score: float = Field(0.0, ge=0, description="Positive markers score")


class BlockBScores(BaseModel):
    """Block B (generation) ELO scores."""

    generation_elo: float = Field(1500, description="Free generation ELO")
    adversarial_elo: float = Field(1500, description="Adversarial ELO")
    long_context_elo: float = Field(1500, description="Long context ELO")


class BlockVScores(BaseModel):
    """Block V (automatic metrics) scores."""

    fertility_rate: float = Field(..., ge=1.0, description="Tokens per word (lower=better)")
    positive_markers: float = Field(..., ge=0, description="Markers per 1K tokens")
    russism_rate: float = Field(..., ge=0, description="Russisms per 1K tokens")
    anglicism_rate: float = Field(..., ge=0, description="Anglicisms per 1K tokens")


class ModelScoreData(BaseModel):
    """Complete score breakdown for a model."""

    elo_rating: float = Field(..., description="Overall ELO rating")
    block_a: BlockAScores = Field(..., description="Block A scores")
    block_b: BlockBScores = Field(..., description="Block B ELO scores")
    block_v: BlockVScores = Field(..., description="Block V metrics")
    badge: Badge = Field(Badge.NONE, description="Quality badge")


class EvaluationMetadataData(BaseModel):
    """Evaluation metadata."""

    benchmark_version: str = Field(..., description="lite/base/large")
    dataset_hash: str = Field(..., description="SHA-256 hash")
    judge_id: str = Field(..., description="Judge model ID")
    judge_calibration_score: float = Field(..., ge=0, le=1, description="Judge calibration")
    total_prompts: int = Field(..., ge=0, description="Prompts evaluated")
    total_comparisons: int = Field(..., ge=0, description="Comparisons made")
    runtime_minutes: float = Field(..., ge=0, description="Runtime in minutes")
    total_cost_usd: float = Field(0.0, ge=0, description="Total cost")
    timestamp: datetime = Field(default_factory=datetime.now, description="Completion time")


class EvaluationResultData(BaseModel):
    """Complete evaluation result for a model."""

    model_id: str = Field(..., description="Model identifier")
    scores: ModelScoreData = Field(..., description="All scores")
    metadata: EvaluationMetadataData = Field(..., description="Metadata")
    comparisons_count: int = Field(0, ge=0, description="Number of comparisons")
    checkpoints: list[str] = Field(default_factory=list, description="Checkpoint file paths")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationResultData:
        """Create from dictionary."""
        return cls.model_validate(data)


# ============================================================================
# Calibration Results
# ============================================================================


class CalibrationResultData(BaseModel):
    """Judge calibration results."""

    judge_id: str = Field(..., description="Judge model ID")
    passed: bool = Field(..., description="Whether calibration passed")

    # Individual metrics
    mc_accuracy: float = Field(..., ge=0, le=1)
    gec_f1: float = Field(..., ge=0, le=1)
    russism_f1: float = Field(..., ge=0, le=1)
    false_positive_rate: float = Field(..., ge=0, le=1)
    pairwise_consistency: float = Field(..., ge=0, le=1)
    position_bias: float = Field(..., ge=0, le=1)
    length_bias_correlation: float = Field(..., ge=-1, le=1)

    # Final score
    final_score: float = Field(..., ge=0, le=1)
    failure_reasons: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)


# Ensure forward references are resolved
GECTask.model_rebuild()
