"""Judge prompt templates for Ukrainian language quality evaluation.

Provides structured prompts for:
- Pairwise comparison of Ukrainian text quality
- Grammar error correction evaluation
- Russism/anglicism detection
- Multiple choice validation
- False positive testing
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class PromptType(str, Enum):
    """Types of judge prompts."""

    PAIRWISE = "pairwise"
    GEC = "gec"
    RUSSISM = "russism"
    MULTIPLE_CHOICE = "multiple_choice"
    FALSE_POSITIVE = "false_positive"
    POSITIVE_MARKERS = "positive_markers"


@dataclass
class PromptTemplate:
    """Template for judge prompts."""

    name: str
    prompt_type: PromptType
    system_prompt: str
    user_template: str
    response_format: dict[str, Any]


# =============================================================================
# System Prompts
# =============================================================================

JUDGE_SYSTEM_BASE = """Ви — експерт з оцінки якості української мови.
Ваше завдання — об'єктивно оцінювати тексти українською мовою.

Ключові критерії оцінки:
1. Нативність — наскільки текст звучить як написаний носієм мови
2. Граматична правильність — відсутність помилок
3. Відсутність русизмів — уникання кальок з російської мови
4. Відсутність англіцизмів — уникання невиправданих запозичень
5. Стилістична доречність — відповідність контексту

Відповідайте ВИКЛЮЧНО у форматі JSON без додаткового тексту."""

PAIRWISE_SYSTEM = f"""{JUDGE_SYSTEM_BASE}

Для порівняння двох текстів оцініть:
- Який текст звучить більш природно українською
- Який має менше граматичних помилок
- Який краще уникає русизмів та англіцизмів
- Який стилістично доречніший"""

GEC_SYSTEM = f"""{JUDGE_SYSTEM_BASE}

Для оцінки виправлення помилок:
- Визначте, чи правильно виправлено помилки
- Оцініть повноту виправлень
- Перевірте, чи не внесено нових помилок"""

RUSSISM_SYSTEM = f"""{JUDGE_SYSTEM_BASE}

Для виявлення русизмів:
- Шукайте лексичні кальки з російської
- Виявляйте синтаксичні русизми
- Відзначайте морфологічні помилки під впливом російської
- Не позначайте прийнятні інтернаціоналізми як русизми"""

MC_SYSTEM = f"""{JUDGE_SYSTEM_BASE}

Для відповіді на тестове питання:
- Уважно прочитайте питання
- Проаналізуйте всі варіанти відповідей
- Оберіть найбільш правильну відповідь
- Обґрунтуйте свій вибір"""

FALSE_POSITIVE_SYSTEM = f"""{JUDGE_SYSTEM_BASE}

Для перевірки тексту на помилки:
- Це текст від визнаного автора або з авторитетного джерела
- Будьте обережні з позначенням помилок
- Враховуйте контекст та стиль
- Деякі архаїзми або діалектизми можуть бути навмисними"""


# =============================================================================
# User Prompt Templates
# =============================================================================

PAIRWISE_USER_TEMPLATE = """Порівняйте два тексти українською мовою та визначте, який з них кращий.

Промпт: {prompt}

Текст A:
{response_a}

Текст B:
{response_b}

Оцініть за критеріями:
1. Нативність (наскільки природно звучить)
2. Граматична правильність
3. Відсутність русизмів
4. Стилістика

Відповідь у форматі JSON:
{{
  "winner": "A" | "B" | "tie",
  "confidence": "high" | "medium" | "low",
  "reasoning": "коротке пояснення (1-2 речення)",
  "scores": {{
    "a": {{"nativeness": 1-5, "grammar": 1-5, "russisms": 1-5, "style": 1-5}},
    "b": {{"nativeness": 1-5, "grammar": 1-5, "russisms": 1-5, "style": 1-5}}
  }}
}}"""

GEC_USER_TEMPLATE = """Оцініть якість виправлення граматичних помилок.

Оригінальний текст (з помилками):
{original}

Виправлений текст:
{corrected}

Еталонне виправлення:
{reference}

Визначте:
1. Чи правильно виправлено помилки
2. Чи є пропущені виправлення
3. Чи внесено зайві зміни

Відповідь у форматі JSON:
{{
  "correct_fixes": ["список правильних виправлень"],
  "missed_fixes": ["список пропущених виправлень"],
  "wrong_fixes": ["список неправильних виправлень"],
  "precision": 0.0-1.0,
  "recall": 0.0-1.0,
  "f1": 0.0-1.0
}}"""

RUSSISM_USER_TEMPLATE = """Проаналізуйте текст на наявність русизмів.

Текст:
{text}

Знайдіть усі русизми та кальки з російської мови.

Відповідь у форматі JSON:
{{
  "russisms": [
    {{
      "text": "знайдений русизм",
      "position": [start, end],
      "correction": "правильний варіант",
      "category": "lexical|syntactic|morphological",
      "severity": "critical|high|medium|low"
    }}
  ],
  "total_count": число,
  "severity_breakdown": {{"critical": n, "high": n, "medium": n, "low": n}}
}}"""

MC_USER_TEMPLATE = """Дайте відповідь на тестове питання з української мови.

Питання:
{question}

Варіанти відповідей:
{options}

Оберіть правильну відповідь та поясніть свій вибір.

Відповідь у форматі JSON:
{{
  "answer": "A" | "B" | "C" | "D",
  "confidence": "high" | "medium" | "low",
  "reasoning": "пояснення вибору"
}}"""

FALSE_POSITIVE_USER_TEMPLATE = """Перевірте текст на наявність мовних помилок.

Текст:
{text}

Джерело: {source}
{author_note}

Чи є в цьому тексті справжні мовні помилки?
Пам'ятайте: текст від визнаного автора може містити навмисні стилістичні особливості.

Відповідь у форматі JSON:
{{
  "has_errors": true | false,
  "errors": [
    {{
      "text": "помилка",
      "position": [start, end],
      "type": "тип помилки",
      "is_intentional": true | false,
      "explanation": "пояснення"
    }}
  ],
  "false_positive_risk": "high" | "medium" | "low",
  "reasoning": "загальне пояснення"
}}"""

POSITIVE_MARKERS_USER_TEMPLATE = """Знайдіть позитивні маркери нативної української мови в тексті.

Текст:
{text}

Шукайте:
- Кличний відмінок (Пане Іване, друже)
- Частки (же, бо, адже, хіба, невже)
- Питомі сполучники (проте, однак, утім, зате, отже)
- Пестливі форми та зменшувальні суфікси

Відповідь у форматі JSON:
{{
  "markers": [
    {{
      "text": "знайдений маркер",
      "position": [start, end],
      "category": "vocative|particle|conjunction|diminutive|other",
      "weight": 0.5-2.0
    }}
  ],
  "total_count": число,
  "nativeness_score": 0-100,
  "category_breakdown": {{"vocative": n, "particle": n, ...}}
}}"""


# =============================================================================
# Response Format Schemas
# =============================================================================

PAIRWISE_RESPONSE_FORMAT = {
    "type": "object",
    "properties": {
        "winner": {"type": "string", "enum": ["A", "B", "tie"]},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "reasoning": {"type": "string"},
        "scores": {
            "type": "object",
            "properties": {
                "a": {
                    "type": "object",
                    "properties": {
                        "nativeness": {"type": "integer", "minimum": 1, "maximum": 5},
                        "grammar": {"type": "integer", "minimum": 1, "maximum": 5},
                        "russisms": {"type": "integer", "minimum": 1, "maximum": 5},
                        "style": {"type": "integer", "minimum": 1, "maximum": 5},
                    },
                },
                "b": {
                    "type": "object",
                    "properties": {
                        "nativeness": {"type": "integer", "minimum": 1, "maximum": 5},
                        "grammar": {"type": "integer", "minimum": 1, "maximum": 5},
                        "russisms": {"type": "integer", "minimum": 1, "maximum": 5},
                        "style": {"type": "integer", "minimum": 1, "maximum": 5},
                    },
                },
            },
        },
    },
    "required": ["winner", "confidence", "reasoning"],
}

GEC_RESPONSE_FORMAT = {
    "type": "object",
    "properties": {
        "correct_fixes": {"type": "array", "items": {"type": "string"}},
        "missed_fixes": {"type": "array", "items": {"type": "string"}},
        "wrong_fixes": {"type": "array", "items": {"type": "string"}},
        "precision": {"type": "number", "minimum": 0, "maximum": 1},
        "recall": {"type": "number", "minimum": 0, "maximum": 1},
        "f1": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["precision", "recall", "f1"],
}

MC_RESPONSE_FORMAT = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "enum": ["A", "B", "C", "D", "E"]},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "reasoning": {"type": "string"},
    },
    "required": ["answer", "confidence"],
}

RUSSISM_RESPONSE_FORMAT = {
    "type": "object",
    "properties": {
        "russisms": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "position": {"type": "array", "items": {"type": "integer"}},
                    "correction": {"type": "string"},
                    "category": {"type": "string"},
                    "severity": {"type": "string"},
                },
            },
        },
        "total_count": {"type": "integer"},
        "severity_breakdown": {"type": "object"},
    },
    "required": ["russisms", "total_count"],
}


# =============================================================================
# Template Registry
# =============================================================================

PROMPT_TEMPLATES: dict[PromptType, PromptTemplate] = {
    PromptType.PAIRWISE: PromptTemplate(
        name="pairwise_comparison",
        prompt_type=PromptType.PAIRWISE,
        system_prompt=PAIRWISE_SYSTEM,
        user_template=PAIRWISE_USER_TEMPLATE,
        response_format=PAIRWISE_RESPONSE_FORMAT,
    ),
    PromptType.GEC: PromptTemplate(
        name="gec_evaluation",
        prompt_type=PromptType.GEC,
        system_prompt=GEC_SYSTEM,
        user_template=GEC_USER_TEMPLATE,
        response_format=GEC_RESPONSE_FORMAT,
    ),
    PromptType.RUSSISM: PromptTemplate(
        name="russism_detection",
        prompt_type=PromptType.RUSSISM,
        system_prompt=RUSSISM_SYSTEM,
        user_template=RUSSISM_USER_TEMPLATE,
        response_format=RUSSISM_RESPONSE_FORMAT,
    ),
    PromptType.MULTIPLE_CHOICE: PromptTemplate(
        name="multiple_choice",
        prompt_type=PromptType.MULTIPLE_CHOICE,
        system_prompt=MC_SYSTEM,
        user_template=MC_USER_TEMPLATE,
        response_format=MC_RESPONSE_FORMAT,
    ),
    PromptType.FALSE_POSITIVE: PromptTemplate(
        name="false_positive_check",
        prompt_type=PromptType.FALSE_POSITIVE,
        system_prompt=FALSE_POSITIVE_SYSTEM,
        user_template=FALSE_POSITIVE_USER_TEMPLATE,
        response_format={},  # Variable format
    ),
    PromptType.POSITIVE_MARKERS: PromptTemplate(
        name="positive_markers",
        prompt_type=PromptType.POSITIVE_MARKERS,
        system_prompt=JUDGE_SYSTEM_BASE,
        user_template=POSITIVE_MARKERS_USER_TEMPLATE,
        response_format={},  # Variable format
    ),
}


def get_template(prompt_type: PromptType) -> PromptTemplate:
    """Get prompt template by type.

    Args:
        prompt_type: Type of prompt to retrieve.

    Returns:
        Corresponding prompt template.

    Raises:
        KeyError: If prompt type not found.
    """
    return PROMPT_TEMPLATES[prompt_type]


def format_pairwise_prompt(
    prompt: str,
    response_a: str,
    response_b: str,
) -> tuple[str, str]:
    """Format a pairwise comparison prompt.

    Args:
        prompt: Original prompt/question.
        response_a: First response.
        response_b: Second response.

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    template = PROMPT_TEMPLATES[PromptType.PAIRWISE]
    user_prompt = template.user_template.format(
        prompt=prompt,
        response_a=response_a,
        response_b=response_b,
    )
    return template.system_prompt, user_prompt


def format_gec_prompt(
    original: str,
    corrected: str,
    reference: str,
) -> tuple[str, str]:
    """Format a GEC evaluation prompt.

    Args:
        original: Original text with errors.
        corrected: Model's correction.
        reference: Reference correction.

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    template = PROMPT_TEMPLATES[PromptType.GEC]
    user_prompt = template.user_template.format(
        original=original,
        corrected=corrected,
        reference=reference,
    )
    return template.system_prompt, user_prompt


def format_russism_prompt(text: str) -> tuple[str, str]:
    """Format a russism detection prompt.

    Args:
        text: Text to analyze.

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    template = PROMPT_TEMPLATES[PromptType.RUSSISM]
    user_prompt = template.user_template.format(text=text)
    return template.system_prompt, user_prompt


def format_mc_prompt(
    question: str,
    options: list[str],
) -> tuple[str, str]:
    """Format a multiple choice prompt.

    Args:
        question: The question text.
        options: List of answer options.

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    template = PROMPT_TEMPLATES[PromptType.MULTIPLE_CHOICE]
    options_text = "\n".join(options)
    user_prompt = template.user_template.format(
        question=question,
        options=options_text,
    )
    return template.system_prompt, user_prompt


def format_false_positive_prompt(
    text: str,
    source: str = "Unknown",
    author: str | None = None,
) -> tuple[str, str]:
    """Format a false positive check prompt.

    Args:
        text: Text to check.
        source: Source of the text.
        author: Optional author name.

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    template = PROMPT_TEMPLATES[PromptType.FALSE_POSITIVE]
    author_note = f"Автор: {author}" if author else ""
    user_prompt = template.user_template.format(
        text=text,
        source=source,
        author_note=author_note,
    )
    return template.system_prompt, user_prompt
