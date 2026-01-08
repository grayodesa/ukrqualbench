"""ZNO (Ukrainian External Independent Evaluation) dataset loader.

Loads multiple choice questions from the ZNO Ukrainian language dataset:
https://huggingface.co/datasets/INSAIT-Institute/zno_ukr

License: MIT
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

from ukrqualbench.core.schemas import MultipleChoiceTask, TaskDifficulty

# ZNO subject to category mapping
SUBJECT_CATEGORY_MAPPING: dict[str, str] = {
    "ukrainian_language": "grammar",
    "ukrainian_literature": "literature",
}

# Topic to subcategory mapping
TOPIC_MAPPING: dict[str, str] = {
    "орфографія": "orthography",
    "пунктуація": "punctuation",
    "лексика": "lexicon",
    "морфологія": "morphology",
    "синтаксис": "syntax",
    "стилістика": "stylistics",
    "фонетика": "phonetics",
    "орфоепія": "orthoepy",
    "граматика": "grammar",
}


@dataclass
class ZNORawQuestion:
    """Raw question from ZNO dataset."""

    question_id: str
    question_text: str
    options: list[str]
    correct_answer: str
    subject: str
    topic: str | None = None
    year: int | None = None
    explanation: str | None = None


class ZNOLoader:
    """Loader for ZNO (ЗНО) Ukrainian language dataset.

    The ZNO dataset contains multiple choice questions from
    Ukrainian standardized tests for grammar, orthography,
    punctuation, and literature.

    Expected formats:
    1. HuggingFace JSON format
    2. Custom JSON format with questions array
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        """Initialize loader.

        Args:
            data_dir: Path to ZNO dataset directory.
        """
        self.data_dir = data_dir

    def load_from_json(self, json_path: Path) -> list[ZNORawQuestion]:
        """Load questions from JSON file.

        Args:
            json_path: Path to JSON file.

        Returns:
            List of raw ZNO questions.
        """
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        questions: list[ZNORawQuestion] = []

        # Handle different JSON formats
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict) and "data" in data:
            items = data["data"]
        elif isinstance(data, dict) and "questions" in data:
            items = data["questions"]
        else:
            raise ValueError(f"Unknown JSON format in {json_path}")

        for i, item in enumerate(items):
            question = self._parse_item(item, i)
            if question:
                questions.append(question)

        return questions

    def load_from_jsonl(self, jsonl_path: Path) -> list[ZNORawQuestion]:
        """Load questions from JSONL file (HuggingFace format).

        Args:
            jsonl_path: Path to JSONL file.

        Returns:
            List of raw ZNO questions.
        """
        questions: list[ZNORawQuestion] = []

        with open(jsonl_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                item = json.loads(line)
                question = self._parse_item(item, i)
                if question:
                    questions.append(question)

        return questions

    def _parse_item(self, item: dict[str, Any], index: int) -> ZNORawQuestion | None:
        """Parse a single item into ZNORawQuestion.

        Args:
            item: Dictionary from JSON/JSONL.
            index: Item index for ID generation.

        Returns:
            Parsed question or None if invalid.
        """
        # Try different field names
        question_text = (
            item.get("question")
            or item.get("question_text")
            or item.get("text")
            or item.get("prompt")
        )

        if not question_text:
            return None

        # Parse options
        options = self._parse_options(item)
        if not options or len(options) < 2:
            return None

        # Parse correct answer
        correct = self._parse_correct_answer(item, options)
        if not correct:
            return None

        # Generate ID
        question_id = item.get("id") or item.get("question_id") or f"zno_{index:05d}"

        return ZNORawQuestion(
            question_id=str(question_id),
            question_text=question_text,
            options=options,
            correct_answer=correct,
            subject=item.get("subject", "ukrainian_language"),
            topic=item.get("topic") or item.get("category"),
            year=item.get("year"),
            explanation=item.get("explanation") or item.get("rationale"),
        )

    def _parse_options(self, item: dict[str, Any]) -> list[str]:
        """Parse answer options from various formats."""
        # Direct options list
        if "options" in item and isinstance(item["options"], list):
            return item["options"]

        # Choices field
        if "choices" in item and isinstance(item["choices"], list):
            return item["choices"]

        # Lettered options (A, B, C, D)
        options = []
        for letter in ["A", "B", "C", "D", "E"]:
            if letter in item:
                options.append(item[letter])
            elif f"option_{letter}" in item:
                options.append(item[f"option_{letter}"])
            elif f"answer_{letter.lower()}" in item:
                options.append(item[f"answer_{letter.lower()}"])

        return options

    def _parse_correct_answer(
        self, item: dict[str, Any], options: list[str]
    ) -> str | None:
        """Parse correct answer from various formats."""
        # Direct answer field - check each field explicitly to handle 0 index
        answer = item.get("answer")
        if answer is None:
            answer = item.get("correct")
        if answer is None:
            answer = item.get("correct_answer")

        if answer is None:
            return None

        # If answer is an index
        if isinstance(answer, int):
            if 0 <= answer < len(options):
                return chr(ord("A") + answer)
            return None

        # If answer is a letter
        answer_str = str(answer).upper().strip()
        if len(answer_str) == 1 and answer_str in "ABCDE":
            return answer_str

        # If answer is the actual text
        for i, opt in enumerate(options):
            if opt.strip().lower() == answer_str.lower():
                return chr(ord("A") + i)

        return None

    def convert_to_tasks(
        self,
        questions: list[ZNORawQuestion],
        category_filter: str | None = None,
        max_tasks: int | None = None,
    ) -> list[MultipleChoiceTask]:
        """Convert ZNO questions to MultipleChoiceTask schema.

        Args:
            questions: List of raw ZNO questions.
            category_filter: Only include specific category.
            max_tasks: Maximum number of tasks to return.

        Returns:
            List of multiple choice tasks.
        """
        tasks: list[MultipleChoiceTask] = []

        for q in questions:
            # Map category
            category = self._map_category(q)
            if category_filter and category != category_filter:
                continue

            # Format options with letters
            formatted_options = [
                f"{chr(ord('A') + i)}) {opt}"
                for i, opt in enumerate(q.options)
            ]

            task = MultipleChoiceTask(
                id=q.question_id,
                category=category,
                subcategory=self._map_subcategory(q),
                prompt=q.question_text,
                options=formatted_options,
                correct=q.correct_answer,
                explanation=q.explanation,
                difficulty=self._determine_difficulty(q),
                source=f"ZNO-{q.year}" if q.year else "ZNO",
            )
            tasks.append(task)

            if max_tasks and len(tasks) >= max_tasks:
                break

        return tasks

    def _map_category(self, question: ZNORawQuestion) -> str:
        """Map question subject/topic to category."""
        # Check topic first
        if question.topic:
            topic_lower = question.topic.lower()
            for key, value in TOPIC_MAPPING.items():
                if key in topic_lower:
                    return value

        # Fall back to subject
        return SUBJECT_CATEGORY_MAPPING.get(question.subject, "grammar")

    def _map_subcategory(self, question: ZNORawQuestion) -> str | None:
        """Map question topic to subcategory."""
        if not question.topic:
            return None

        topic_lower = question.topic.lower()
        for key, value in TOPIC_MAPPING.items():
            if key in topic_lower:
                return value

        return question.topic

    def _determine_difficulty(self, question: ZNORawQuestion) -> TaskDifficulty:
        """Determine task difficulty based on question characteristics."""
        # More options = harder
        if len(question.options) >= 5:
            return TaskDifficulty.HARD

        # Complex topics are harder
        hard_topics = {"синтаксис", "стилістика", "пунктуація"}
        if question.topic and any(t in question.topic.lower() for t in hard_topics):
            return TaskDifficulty.HARD

        # Longer questions tend to be harder
        if len(question.question_text) > 300:
            return TaskDifficulty.HARD
        if len(question.question_text) > 150:
            return TaskDifficulty.MEDIUM

        return TaskDifficulty.EASY

    def iter_by_category(
        self, questions: list[ZNORawQuestion], category: str
    ) -> Iterator[MultipleChoiceTask]:
        """Iterate over tasks filtered by category.

        Args:
            questions: List of raw ZNO questions.
            category: Category to filter by.

        Yields:
            Multiple choice tasks.
        """
        yield from self.convert_to_tasks(questions, category_filter=category)

    def get_statistics(self, questions: list[ZNORawQuestion]) -> dict[str, int]:
        """Get statistics about loaded questions.

        Args:
            questions: List of questions.

        Returns:
            Dictionary with counts by category.
        """
        stats: dict[str, int] = {"total": len(questions)}

        for q in questions:
            category = self._map_category(q)
            stats[category] = stats.get(category, 0) + 1

        return stats
