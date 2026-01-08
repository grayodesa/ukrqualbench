"""Base judge class for LLM-based evaluation.

Provides abstract interface for judge implementations:
- Pairwise comparison
- GEC evaluation
- Multiple choice assessment
- Russism detection
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ukrqualbench.core.schemas import (
    ConfidenceLevel,
    JudgeVerdictData,
    WinnerChoice,
)
from ukrqualbench.judges.prompts import PromptTemplate, PromptType, get_template

if TYPE_CHECKING:
    from collections.abc import Sequence


@runtime_checkable
class ModelClient(Protocol):
    """Protocol for LLM model clients.

    Any model client must implement this interface to work with judges.
    """

    @property
    def model_id(self) -> str:
        """Return the model identifier."""
        ...

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> ModelResponse:
        """Generate a response from the model.

        Args:
            prompt: User prompt.
            system_prompt: Optional system prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            json_mode: Whether to enforce JSON output.

        Returns:
            Model response with text, tokens, and latency.
        """
        ...


@dataclass
class ModelResponse:
    """Response from a model client."""

    text: str
    tokens_used: int
    latency_ms: float
    model_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    cost_usd: float = 0.0


@dataclass
class JudgeConfig:
    """Configuration for judge behavior."""

    temperature: float = 0.0
    max_tokens: int = 1024
    json_mode: bool = True
    max_retries: int = 3
    retry_delay: float = 1.0
    strict_json: bool = True


class JSONParseError(Exception):
    """Raised when JSON parsing fails."""

    def __init__(self, message: str, raw_response: str) -> None:
        super().__init__(message)
        self.raw_response = raw_response


class BaseJudge(ABC):
    """Abstract base class for LLM judges.

    Judges evaluate Ukrainian language quality using LLM-as-a-Judge pattern.

    Example:
        >>> judge = PairwiseJudge(model_client)
        >>> verdict = await judge.judge_pairwise(prompt, response_a, response_b)
        >>> verdict.winner
        'A'
    """

    def __init__(
        self,
        model: ModelClient,
        config: JudgeConfig | None = None,
    ) -> None:
        """Initialize judge with model client.

        Args:
            model: Model client implementing ModelClient protocol.
            config: Optional judge configuration.
        """
        self._model = model
        self._config = config or JudgeConfig()
        self._call_count = 0
        self._total_tokens = 0
        self._total_latency_ms = 0.0

    @property
    def model_id(self) -> str:
        """Return the judge model ID."""
        return self._model.model_id

    @property
    def call_count(self) -> int:
        """Return number of judge calls made."""
        return self._call_count

    @property
    def total_tokens(self) -> int:
        """Return total tokens used."""
        return self._total_tokens

    @property
    def average_latency_ms(self) -> float:
        """Return average latency per call."""
        if self._call_count == 0:
            return 0.0
        return self._total_latency_ms / self._call_count

    def get_template(self, prompt_type: PromptType) -> PromptTemplate:
        """Get prompt template by type.

        Args:
            prompt_type: Type of prompt to retrieve.

        Returns:
            Corresponding prompt template.
        """
        return get_template(prompt_type)

    async def _call_model(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> ModelResponse:
        """Call the model with given prompts.

        Args:
            system_prompt: System prompt.
            user_prompt: User prompt.

        Returns:
            Model response.

        Raises:
            RuntimeError: If all retries fail.
        """
        last_error: Exception | None = None

        for attempt in range(self._config.max_retries):
            try:
                response = await self._model.generate(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    temperature=self._config.temperature,
                    max_tokens=self._config.max_tokens,
                    json_mode=self._config.json_mode,
                )

                # Update statistics
                self._call_count += 1
                self._total_tokens += response.tokens_used
                self._total_latency_ms += response.latency_ms

                return response

            except Exception as e:
                last_error = e
                if attempt < self._config.max_retries - 1:
                    import asyncio
                    await asyncio.sleep(self._config.retry_delay * (attempt + 1))

        raise RuntimeError(
            f"All {self._config.max_retries} retries failed"
        ) from last_error

    def _parse_json_response(
        self,
        text: str,
        required_fields: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Parse JSON from model response.

        Handles various JSON formats including markdown code blocks.

        Args:
            text: Raw response text.
            required_fields: Optional list of required field names.

        Returns:
            Parsed JSON as dictionary.

        Raises:
            JSONParseError: If parsing fails or required fields missing.
        """
        # Try to extract JSON from markdown code blocks
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if json_match:
            text = json_match.group(1)

        # Try to find JSON object
        json_obj_match = re.search(r"\{[\s\S]*\}", text)
        if json_obj_match:
            text = json_obj_match.group(0)

        try:
            result = json.loads(text)
        except json.JSONDecodeError as e:
            raise JSONParseError(
                f"Failed to parse JSON: {e}",
                raw_response=text,
            ) from e

        if not isinstance(result, dict):
            raise JSONParseError(
                f"Expected JSON object, got {type(result).__name__}",
                raw_response=text,
            )

        # Validate required fields
        if required_fields:
            missing = [f for f in required_fields if f not in result]
            if missing:
                raise JSONParseError(
                    f"Missing required fields: {missing}",
                    raw_response=text,
                )

        return result

    def _parse_winner(self, value: str) -> WinnerChoice:
        """Parse winner choice from string.

        Args:
            value: Winner value from JSON.

        Returns:
            Normalized WinnerChoice enum.
        """
        value_upper = str(value).upper().strip()

        if value_upper in ("A", "1", "FIRST", "MODEL_A"):
            return WinnerChoice.A
        if value_upper in ("B", "2", "SECOND", "MODEL_B"):
            return WinnerChoice.B
        if value_upper in ("TIE", "DRAW", "EQUAL", "BOTH", "NEITHER"):
            return WinnerChoice.TIE

        # Default to tie for ambiguous cases
        return WinnerChoice.TIE

    def _parse_confidence(self, value: str) -> ConfidenceLevel:
        """Parse confidence level from string.

        Args:
            value: Confidence value from JSON.

        Returns:
            Normalized ConfidenceLevel enum.
        """
        value_lower = str(value).lower().strip()

        if value_lower in ("high", "висока", "впевнено"):
            return ConfidenceLevel.HIGH
        if value_lower in ("low", "низька", "невпевнено"):
            return ConfidenceLevel.LOW

        # Default to medium
        return ConfidenceLevel.MEDIUM

    @abstractmethod
    async def evaluate(self, **kwargs: Any) -> Any:
        """Perform evaluation.

        Subclasses must implement this method for their specific
        evaluation type (pairwise, GEC, MC, etc.).

        Args:
            **kwargs: Evaluation-specific arguments.

        Returns:
            Evaluation result (type depends on implementation).
        """
        ...

    def reset_statistics(self) -> None:
        """Reset usage statistics."""
        self._call_count = 0
        self._total_tokens = 0
        self._total_latency_ms = 0.0


class PairwiseJudgeBase(BaseJudge):
    """Base class for pairwise comparison judges.

    Evaluates two responses and determines which is better
    based on Ukrainian language quality criteria.
    """

    def _create_verdict(
        self,
        parsed: dict[str, Any],
        raw_response: str,
        latency_ms: float,
    ) -> JudgeVerdictData:
        """Create verdict from parsed JSON response.

        Args:
            parsed: Parsed JSON response.
            raw_response: Raw response text.
            latency_ms: Response latency.

        Returns:
            Structured verdict data.
        """
        return JudgeVerdictData(
            winner=self._parse_winner(parsed.get("winner", "tie")),
            confidence=self._parse_confidence(parsed.get("confidence", "medium")),
            reasoning=parsed.get("reasoning", ""),
            raw_response=raw_response,
            latency_ms=latency_ms,
        )

    async def evaluate(self, **kwargs: Any) -> JudgeVerdictData:
        """Evaluate two responses in pairwise comparison.

        Kwargs:
            prompt: Original prompt/question.
            response_a: First response.
            response_b: Second response.

        Returns:
            Judge verdict with winner, confidence, and reasoning.
        """
        prompt = kwargs.get("prompt", "")
        response_a = kwargs.get("response_a", "")
        response_b = kwargs.get("response_b", "")
        return await self.judge_pairwise(prompt, response_a, response_b)

    @abstractmethod
    async def judge_pairwise(
        self,
        prompt: str,
        response_a: str,
        response_b: str,
    ) -> JudgeVerdictData:
        """Judge two responses in pairwise comparison.

        Args:
            prompt: Original prompt/question.
            response_a: First response.
            response_b: Second response.

        Returns:
            Judge verdict with winner, confidence, and reasoning.
        """
        ...


class GECJudgeBase(BaseJudge):
    """Base class for GEC (Grammar Error Correction) judges."""

    @abstractmethod
    async def evaluate_correction(
        self,
        original: str,
        corrected: str,
        reference: str,
    ) -> dict[str, Any]:
        """Evaluate a grammar correction.

        Args:
            original: Text with errors.
            corrected: Model's correction.
            reference: Reference correction.

        Returns:
            Dictionary with precision, recall, F1, and details.
        """
        ...


class MultipleChoiceJudgeBase(BaseJudge):
    """Base class for multiple choice judges."""

    @abstractmethod
    async def evaluate_answer(
        self,
        question: str,
        options: list[str],
        model_answer: str,
        correct_answer: str,
    ) -> dict[str, Any]:
        """Evaluate a multiple choice answer.

        Args:
            question: Question text.
            options: Answer options.
            model_answer: Model's selected answer.
            correct_answer: Correct answer.

        Returns:
            Dictionary with correctness and analysis.
        """
        ...


class RussismJudgeBase(BaseJudge):
    """Base class for russism detection judges."""

    @abstractmethod
    async def detect_russisms(
        self,
        text: str,
    ) -> dict[str, Any]:
        """Detect russisms in text.

        Args:
            text: Text to analyze.

        Returns:
            Dictionary with russisms found and analysis.
        """
        ...
