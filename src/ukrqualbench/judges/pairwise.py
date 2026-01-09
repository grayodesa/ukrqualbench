"""Pairwise comparison judge for Ukrainian language quality.

Implements LLM-as-a-Judge pattern for comparing two Ukrainian text responses
and determining which exhibits better language quality.

Key features:
- Position bias mitigation through response shuffling
- Detailed score breakdown (nativeness, grammar, russisms, style)
- Consistency checking with repeated evaluations
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ukrqualbench.core.schemas import (
    ConfidenceLevel,
    JudgeVerdictData,
    PositionOrder,
    WinnerChoice,
)
from ukrqualbench.judges.base import (
    JSONParseError,
    JudgeConfig,
    ModelClient,
    PairwiseJudgeBase,
)
from ukrqualbench.judges.prompts import (
    format_pairwise_prompt,
)

if TYPE_CHECKING:
    pass


@dataclass
class PairwiseScores:
    """Detailed scores for a response in pairwise comparison."""

    nativeness: int  # 1-5
    grammar: int  # 1-5
    russisms: int  # 1-5 (higher = fewer russisms)
    style: int  # 1-5

    @property
    def total(self) -> int:
        """Calculate total score."""
        return self.nativeness + self.grammar + self.russisms + self.style

    @property
    def average(self) -> float:
        """Calculate average score."""
        return self.total / 4.0


@dataclass
class DetailedVerdict:
    """Extended verdict with detailed scores."""

    verdict: JudgeVerdictData
    scores_a: PairwiseScores | None
    scores_b: PairwiseScores | None
    position_order: PositionOrder


class PairwiseJudge(PairwiseJudgeBase):
    """Judge for pairwise comparison of Ukrainian text quality.

    Compares two responses and determines which exhibits better
    Ukrainian language quality based on:
    - Nativeness (natural, native-sounding Ukrainian)
    - Grammar correctness
    - Absence of russisms
    - Stylistic appropriateness

    Example:
        >>> judge = PairwiseJudge(model_client)
        >>> verdict = await judge.judge_pairwise(
        ...     prompt="Поясніть фотосинтез",
        ...     response_a="Фотосинтез - це процес...",
        ...     response_b="Фотосинтез є процесом..."
        ... )
        >>> verdict.winner
        <WinnerChoice.A: 'A'>
    """

    def __init__(
        self,
        model: ModelClient,
        config: JudgeConfig | None = None,
        shuffle_positions: bool = True,
    ) -> None:
        """Initialize pairwise judge.

        Args:
            model: Model client for judge calls.
            config: Optional judge configuration.
            shuffle_positions: Whether to randomize response positions
                to mitigate position bias.
        """
        super().__init__(model, config)
        self._shuffle_positions = shuffle_positions

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
        detailed = await self.judge_detailed(prompt, response_a, response_b)
        return detailed.verdict

    async def judge_detailed(
        self,
        prompt: str,
        response_a: str,
        response_b: str,
    ) -> DetailedVerdict:
        """Judge with detailed score breakdown.

        Args:
            prompt: Original prompt/question.
            response_a: First response.
            response_b: Second response.

        Returns:
            Detailed verdict with scores for each criterion.
        """
        # Determine position order
        if self._shuffle_positions and random.random() < 0.5:
            # Swap positions
            actual_a, actual_b = response_b, response_a
            position_order = PositionOrder.BA
        else:
            actual_a, actual_b = response_a, response_b
            position_order = PositionOrder.AB

        # Format prompt
        system_prompt, user_prompt = format_pairwise_prompt(
            prompt=prompt,
            response_a=actual_a,
            response_b=actual_b,
        )

        # Call model
        response = await self._call_model(system_prompt, user_prompt)

        # Parse response
        try:
            parsed = self._parse_json_response(
                response.text,
                required_fields=["winner", "confidence"],
            )
        except JSONParseError:
            # Return uncertain verdict on parse failure
            return DetailedVerdict(
                verdict=JudgeVerdictData(
                    winner=WinnerChoice.TIE,
                    confidence=ConfidenceLevel.LOW,
                    reasoning="Failed to parse judge response",
                    raw_response=response.text,
                    latency_ms=response.latency_ms,
                ),
                scores_a=None,
                scores_b=None,
                position_order=position_order,
            )

        # Parse scores if available
        scores_a, scores_b = self._parse_scores(parsed)

        # Map winner back to original positions
        raw_winner = self._parse_winner(parsed.get("winner", "tie"))
        actual_winner = self._map_winner(raw_winner, position_order)

        # Map scores back to original positions
        if position_order == PositionOrder.BA:
            scores_a, scores_b = scores_b, scores_a

        verdict = JudgeVerdictData(
            winner=actual_winner,
            confidence=self._parse_confidence(parsed.get("confidence", "medium")),
            reasoning=parsed.get("reasoning", ""),
            raw_response=response.text,
            latency_ms=response.latency_ms,
        )

        return DetailedVerdict(
            verdict=verdict,
            scores_a=scores_a,
            scores_b=scores_b,
            position_order=position_order,
        )

    async def judge_with_consistency(
        self,
        prompt: str,
        response_a: str,
        response_b: str,
        rounds: int = 3,
    ) -> tuple[JudgeVerdictData, float]:
        """Judge with consistency checking across multiple rounds.

        Performs multiple evaluations and returns the majority verdict
        along with a consistency score.

        Args:
            prompt: Original prompt/question.
            response_a: First response.
            response_b: Second response.
            rounds: Number of evaluation rounds.

        Returns:
            Tuple of (final verdict, consistency score 0-1).
        """
        verdicts: list[DetailedVerdict] = []

        for _ in range(rounds):
            verdict = await self.judge_detailed(prompt, response_a, response_b)
            verdicts.append(verdict)

        # Count winners
        winner_counts: dict[WinnerChoice, int] = {
            WinnerChoice.A: 0,
            WinnerChoice.B: 0,
            WinnerChoice.TIE: 0,
        }
        for v in verdicts:
            winner_counts[v.verdict.winner] += 1

        # Find majority winner
        majority_winner = max(winner_counts, key=lambda k: winner_counts[k])
        majority_count = winner_counts[majority_winner]

        # Calculate consistency (how often judge agreed with itself)
        consistency = majority_count / rounds

        # Use the verdict that matches majority
        final_verdict = next(v.verdict for v in verdicts if v.verdict.winner == majority_winner)

        # Adjust confidence based on consistency
        if consistency < 0.6:
            final_verdict = JudgeVerdictData(
                winner=final_verdict.winner,
                confidence=ConfidenceLevel.LOW,
                reasoning=final_verdict.reasoning,
                raw_response=final_verdict.raw_response,
                latency_ms=final_verdict.latency_ms,
            )

        return final_verdict, consistency

    def _parse_scores(
        self,
        parsed: dict[str, Any],
    ) -> tuple[PairwiseScores | None, PairwiseScores | None]:
        """Parse detailed scores from response.

        Args:
            parsed: Parsed JSON response.

        Returns:
            Tuple of (scores_a, scores_b), either may be None.
        """
        scores = parsed.get("scores", {})
        if not isinstance(scores, dict):
            return None, None

        scores_a = self._extract_scores(scores.get("a", {}))
        scores_b = self._extract_scores(scores.get("b", {}))

        return scores_a, scores_b

    def _extract_scores(self, data: dict[str, Any]) -> PairwiseScores | None:
        """Extract PairwiseScores from dictionary.

        Args:
            data: Score dictionary.

        Returns:
            PairwiseScores or None if invalid.
        """
        if not isinstance(data, dict):
            return None

        try:
            return PairwiseScores(
                nativeness=self._clamp_score(data.get("nativeness", 3)),
                grammar=self._clamp_score(data.get("grammar", 3)),
                russisms=self._clamp_score(data.get("russisms", 3)),
                style=self._clamp_score(data.get("style", 3)),
            )
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _clamp_score(value: Any) -> int:
        """Clamp score to valid range 1-5.

        Args:
            value: Score value.

        Returns:
            Integer score in range [1, 5].
        """
        try:
            score = int(value)
            return max(1, min(5, score))
        except (TypeError, ValueError):
            return 3  # Default middle score

    @staticmethod
    def _map_winner(
        winner: WinnerChoice,
        position_order: PositionOrder,
    ) -> WinnerChoice:
        """Map winner back to original positions.

        Args:
            winner: Winner in presentation order.
            position_order: Order responses were presented.

        Returns:
            Winner in original order.
        """
        if position_order == PositionOrder.AB:
            return winner

        # Positions were swapped (BA), so swap winner
        if winner == WinnerChoice.A:
            return WinnerChoice.B
        if winner == WinnerChoice.B:
            return WinnerChoice.A
        return WinnerChoice.TIE


def create_pairwise_judge(
    model: ModelClient,
    shuffle_positions: bool = True,
    temperature: float = 0.0,
    max_retries: int = 3,
) -> PairwiseJudge:
    """Factory function to create a pairwise judge.

    Args:
        model: Model client.
        shuffle_positions: Whether to randomize positions.
        temperature: Sampling temperature.
        max_retries: Maximum retry attempts.

    Returns:
        Configured PairwiseJudge instance.
    """
    config = JudgeConfig(
        temperature=temperature,
        max_retries=max_retries,
        json_mode=True,
    )
    return PairwiseJudge(
        model=model,
        config=config,
        shuffle_positions=shuffle_positions,
    )
