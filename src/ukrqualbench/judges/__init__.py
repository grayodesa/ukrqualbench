"""LLM judge system for pairwise evaluation.

Judges for Ukrainian language quality evaluation:
- Pairwise comparison for relative quality assessment
- GEC (Grammar Error Correction) evaluation
- Multiple choice assessment
- Russism detection
- Judge calibration against gold standards
"""

from ukrqualbench.judges.base import (
    BaseJudge,
    GECJudgeBase,
    JSONParseError,
    JudgeConfig,
    ModelClient,
    MultipleChoiceJudgeBase,
    PairwiseJudgeBase,
    RussismJudgeBase,
)
from ukrqualbench.models.base import ModelResponse
from ukrqualbench.judges.calibrator import (
    FALSE_POSITIVE_THRESHOLD,
    FINAL_SCORE_THRESHOLD,
    GEC_F1_THRESHOLD,
    LENGTH_BIAS_THRESHOLD,
    MC_ACCURACY_THRESHOLD,
    PAIRWISE_CONSISTENCY_THRESHOLD,
    POSITION_BIAS_THRESHOLD,
    RUSSISM_F1_THRESHOLD,
    CalibrationMetrics,
    CalibrationTask,
    JudgeCalibrator,
    create_calibrator,
)
from ukrqualbench.judges.pairwise import (
    DetailedVerdict,
    PairwiseJudge,
    PairwiseScores,
    create_pairwise_judge,
)
from ukrqualbench.judges.prompts import (
    FALSE_POSITIVE_SYSTEM,
    GEC_RESPONSE_FORMAT,
    GEC_SYSTEM,
    GEC_USER_TEMPLATE,
    JUDGE_SYSTEM_BASE,
    MC_RESPONSE_FORMAT,
    MC_SYSTEM,
    MC_USER_TEMPLATE,
    PAIRWISE_RESPONSE_FORMAT,
    PAIRWISE_SYSTEM,
    PAIRWISE_USER_TEMPLATE,
    POSITIVE_MARKERS_USER_TEMPLATE,
    PROMPT_TEMPLATES,
    RUSSISM_RESPONSE_FORMAT,
    RUSSISM_SYSTEM,
    RUSSISM_USER_TEMPLATE,
    PromptTemplate,
    PromptType,
    format_false_positive_prompt,
    format_gec_prompt,
    format_mc_prompt,
    format_pairwise_prompt,
    format_russism_prompt,
    get_template,
)

__all__ = [
    # Prompts
    "FALSE_POSITIVE_SYSTEM",
    "FALSE_POSITIVE_THRESHOLD",
    "FINAL_SCORE_THRESHOLD",
    "GEC_F1_THRESHOLD",
    "GEC_RESPONSE_FORMAT",
    "GEC_SYSTEM",
    "GEC_USER_TEMPLATE",
    "JUDGE_SYSTEM_BASE",
    "LENGTH_BIAS_THRESHOLD",
    "MC_ACCURACY_THRESHOLD",
    "MC_RESPONSE_FORMAT",
    "MC_SYSTEM",
    "MC_USER_TEMPLATE",
    "PAIRWISE_CONSISTENCY_THRESHOLD",
    "PAIRWISE_RESPONSE_FORMAT",
    "PAIRWISE_SYSTEM",
    "PAIRWISE_USER_TEMPLATE",
    "POSITION_BIAS_THRESHOLD",
    "POSITIVE_MARKERS_USER_TEMPLATE",
    "PROMPT_TEMPLATES",
    "RUSSISM_F1_THRESHOLD",
    "RUSSISM_RESPONSE_FORMAT",
    "RUSSISM_SYSTEM",
    "RUSSISM_USER_TEMPLATE",
    # Base classes and protocols
    "BaseJudge",
    # Calibrator
    "CalibrationMetrics",
    "CalibrationTask",
    # Pairwise judge
    "DetailedVerdict",
    "GECJudgeBase",
    "JSONParseError",
    "JudgeCalibrator",
    "JudgeConfig",
    "ModelClient",
    "ModelResponse",
    "MultipleChoiceJudgeBase",
    "PairwiseJudge",
    "PairwiseJudgeBase",
    "PairwiseScores",
    "PromptTemplate",
    "PromptType",
    "RussismJudgeBase",
    "create_calibrator",
    "create_pairwise_judge",
    "format_false_positive_prompt",
    "format_gec_prompt",
    "format_mc_prompt",
    "format_pairwise_prompt",
    "format_russism_prompt",
    "get_template",
]
