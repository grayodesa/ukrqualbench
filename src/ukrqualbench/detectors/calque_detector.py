"""Judge-based calque detector for Ukrainian text.

Uses LLM judge with few-shot examples to detect Russian calques
(lexical, syntactic, morphological) that dictionary-based detection misses.

This detector is designed for Block C metrics where accuracy matters more
than speed, as it makes API calls to the judge model.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ukrqualbench.detectors.base import (
    DetectionMatch,
    DetectionResult,
    DetectionSeverity,
)

if TYPE_CHECKING:
    from ukrqualbench.models.base import ModelResponse

logger = logging.getLogger(__name__)


@runtime_checkable
class ModelClient(Protocol):
    """Protocol for LLM model clients."""

    @property
    def model_id(self) -> str:
        """Return the model identifier."""
        ...

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> ModelResponse:
        """Generate a response from the model."""
        ...


# =============================================================================
# Enhanced Calque Detection Prompt with Few-Shot Examples
# =============================================================================

CALQUE_DETECTION_SYSTEM = """Ви — експерт з української мови, що спеціалізується на виявленні кальок з російської мови.

## ТИПИ КАЛЬОК

### 1. ЛЕКСИЧНІ КАЛЬКИ (заміна слів)
Приклади:
- "прийняти участь" → "взяти участь"
- "міроприємство" → "захід"
- "являється" → "є"
- "слідуючий" → "наступний"
- "отримати досвід" → "здобути досвід"
- "приймати рішення" → "ухвалювати рішення"
- "в даному випадку" → "у цьому випадку"
- "виключення" → "виняток"
- "заключати договір" → "укладати договір"
- "мати місце" → "відбуватися"
- "в значній мірі" → "значною мірою"
- "приймати міри" → "вживати заходів"

### 2. СИНТАКСИЧНІ КАЛЬКИ (структура речення, керування відмінками)
Приклади:
- "по українськи" → "українською" (неправильний прийменник)
- "згідно закону" → "згідно із законом" (керування відмінком)
- "дякуючи йому" → "завдяки йому" (неправильна калька)
- "по причині" → "через" (неправильний прийменник)
- "по відношенню до" → "щодо, стосовно"
- "в конференції" → "у конференції" (перед приголосним)
- "по питанням" → "з питань" (неправильний прийменник)

### 3. МОРФОЛОГІЧНІ КАЛЬКИ (форма слів)
Приклади:
- "більш красивіша" → "красивіша" (подвійний ступінь порівняння)
- "самий великий" → "найбільший" (калька найвищого ступеня)
- "обоїх" → "обох"

### 4. ФРАЗЕОЛОГІЧНІ КАЛЬКИ (сталі вирази)
Приклади:
- "мати на увазі" → "мати на оці" або "йтися про"
- "приводити до відома" → "доводити до відома"
- "з точки зору" → "з погляду"
- "не дивлячись на" → "попри, незважаючи на"
- "в кінці кінців" → "врешті-решт, зрештою"

## ВАЖЛИВО

1. Шукайте ВСІ кальки в тексті, навіть якщо їх декілька
2. Повертайте ТОЧНІ цитати з тексту (не парафраз)
3. НЕ позначайте як кальки:
   - Прийнятні інтернаціоналізми (комп'ютер, інтернет)
   - Стилістично нейтральні варіанти
   - Терміни без українських аналогів
4. Враховуйте контекст — деякі слова можуть бути правильними в певному контексті

Відповідайте ВИКЛЮЧНО у форматі JSON."""

CALQUE_DETECTION_USER = """Знайдіть ВСІ кальки з російської мови в тексті.

Текст для аналізу:
{text}

Поверніть JSON з переліком знайдених кальок:
{{
  "calques": [
    {{
      "text": "точна цитата кальки з тексту",
      "correction": "правильний український варіант",
      "type": "lexical|syntactic|morphological|phraseological",
      "severity": "critical|high|medium"
    }}
  ],
  "total_count": число,
  "analysis_notes": "короткий коментар щодо загальної якості тексту"
}}

Якщо кальок немає, поверніть: {{"calques": [], "total_count": 0, "analysis_notes": "Текст не містить кальок з російської мови."}}"""


# Severity mapping for calque types
CALQUE_SEVERITY_WEIGHTS = {
    "critical": 2.0,
    "high": 1.5,
    "medium": 1.0,
}

TYPE_TO_SEVERITY = {
    "lexical": DetectionSeverity.HIGH,
    "syntactic": DetectionSeverity.CRITICAL,
    "morphological": DetectionSeverity.MEDIUM,
    "phraseological": DetectionSeverity.HIGH,
}


@dataclass
class CalqueDetectionConfig:
    """Configuration for calque detector."""

    temperature: float = 0.0
    max_tokens: int = 2048
    json_mode: bool = True
    max_retries: int = 2


class JudgeBasedCalqueDetector:
    """LLM judge-based calque detector for Ukrainian text.

    Uses an LLM judge with few-shot examples to detect Russian calques
    that dictionary-based regex patterns miss. Designed for Block C metrics
    where accuracy is more important than speed.

    Example:
        >>> from ukrqualbench.models import create_model_client
        >>> client = create_model_client(config.default_judge, config)
        >>> detector = JudgeBasedCalqueDetector(client)
        >>> result = await detector.detect("Ми прийняли участь в заході.")
        >>> result.count
        1
    """

    def __init__(
        self,
        model_client: ModelClient,
        config: CalqueDetectionConfig | None = None,
    ) -> None:
        """Initialize detector with model client.

        Args:
            model_client: LLM client for judge calls.
            config: Optional detector configuration.
        """
        self._client = model_client
        self._config = config or CalqueDetectionConfig()
        self._call_count = 0
        self._total_tokens = 0

    @property
    def model_id(self) -> str:
        """Return the judge model ID."""
        return self._client.model_id

    @property
    def call_count(self) -> int:
        """Return number of detection calls made."""
        return self._call_count

    async def detect(
        self,
        text: str,
        token_count: int | None = None,
    ) -> DetectionResult:
        """Detect calques in text using LLM judge.

        Args:
            text: Text to analyze.
            token_count: Pre-computed token count (optional).

        Returns:
            DetectionResult with all calques found.
        """
        if not text.strip():
            return DetectionResult(
                text=text,
                matches=[],
                total_tokens=token_count or 0,
            )

        # Format prompt
        user_prompt = CALQUE_DETECTION_USER.format(text=text)

        # Call judge model
        try:
            response = await self._client.generate(
                prompt=user_prompt,
                system_prompt=CALQUE_DETECTION_SYSTEM,
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
                json_mode=self._config.json_mode,
            )
            self._call_count += 1
            self._total_tokens += response.tokens_used

            # Parse response
            result_data = self._parse_response(response.text)
            matches = self._convert_to_matches(result_data, text)

        except Exception as e:
            logger.warning("Calque detection failed: %s", e)
            matches = []

        # Estimate tokens if not provided
        if token_count is None:
            token_count = self._estimate_tokens(text)

        return DetectionResult(
            text=text,
            matches=matches,
            total_tokens=token_count,
            metadata={
                "judge_model": self.model_id,
                "analysis_notes": result_data.get("analysis_notes", "")
                if "result_data" in dir()
                else "",
            },
        )

    def _parse_response(self, response_text: str) -> dict[str, Any]:
        """Parse JSON response from judge.

        Args:
            response_text: Raw response from judge.

        Returns:
            Parsed JSON data.
        """
        # Try to extract JSON from response
        text = response_text.strip()

        # Handle markdown code blocks
        if "```json" in text:
            match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
            if match:
                text = match.group(1)
        elif "```" in text:
            match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
            if match:
                text = match.group(1)

        try:
            data: dict[str, Any] = json.loads(text)
            return data
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON response: %s", text[:200])
            return {"calques": [], "total_count": 0}

    def _convert_to_matches(
        self,
        data: dict[str, Any],
        original_text: str,
    ) -> list[DetectionMatch]:
        """Convert parsed calque data to DetectionMatch objects.

        Args:
            data: Parsed JSON response.
            original_text: Original text for position finding.

        Returns:
            List of DetectionMatch objects.
        """
        matches: list[DetectionMatch] = []
        calques = data.get("calques", [])

        for i, calque in enumerate(calques):
            calque_text = calque.get("text", "")
            if not calque_text:
                continue

            # Find position in original text
            start = original_text.lower().find(calque_text.lower())
            if start == -1:
                # Try partial match
                start = 0
            end = start + len(calque_text)

            # Determine severity
            calque_type = calque.get("type", "lexical")
            severity_str = calque.get("severity", "medium")
            severity = TYPE_TO_SEVERITY.get(calque_type, DetectionSeverity.MEDIUM)
            weight = CALQUE_SEVERITY_WEIGHTS.get(severity_str, 1.0)

            match = DetectionMatch(
                start=start,
                end=end,
                matched_text=calque_text,
                pattern_id=f"calque_{calque_type}_{i:03d}",
                category=calque_type,
                severity=severity,
                correction=calque.get("correction"),
                description=f"Калька з російської ({calque_type})",
                weight=weight,
            )
            matches.append(match)

        return matches

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.

        Simple heuristic: ~1.5 tokens per word for Ukrainian.

        Args:
            text: Text to estimate.

        Returns:
            Estimated token count.
        """
        words = len(text.split())
        return int(words * 1.5)


def create_calque_detector(
    model_client: ModelClient,
    config: CalqueDetectionConfig | None = None,
) -> JudgeBasedCalqueDetector:
    """Factory function to create calque detector.

    Args:
        model_client: LLM client for judge calls.
        config: Optional detector configuration.

    Returns:
        Initialized calque detector.
    """
    return JudgeBasedCalqueDetector(model_client, config)
