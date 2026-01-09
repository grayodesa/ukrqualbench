"""UA-GEC 2.0 dataset loader.

Loads grammar error correction data from the UA-GEC dataset:
https://github.com/grammarly/ua-gec

License: CC BY 4.0
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

from ukrqualbench.core.schemas import (
    ErrorAnnotation,
    ErrorSeverity,
    GECTask,
    TaskDifficulty,
)

# Mapping from UA-GEC error categories to our categories
CATEGORY_MAPPING: dict[str, str] = {
    "Fluency:Calque": "russism",
    "Fluency:Collocation": "collocation",
    "Fluency:Grammar": "grammar",
    "Fluency:Register": "register",
    "Grammar:Case": "grammar_case",
    "Grammar:Gender": "grammar_gender",
    "Grammar:Number": "grammar_number",
    "Grammar:Tense": "grammar_tense",
    "Grammar:Aspect": "grammar_aspect",
    "Grammar:VerbVoice": "grammar_voice",
    "Grammar:PartOfSpeech": "grammar_pos",
    "Grammar:Other": "grammar_other",
    "Punctuation": "punctuation",
    "Spelling": "spelling",
}

# Severity mapping based on error type
SEVERITY_MAPPING: dict[str, ErrorSeverity] = {
    "russism": ErrorSeverity.CRITICAL,
    "collocation": ErrorSeverity.HIGH,
    "grammar_case": ErrorSeverity.HIGH,
    "grammar_gender": ErrorSeverity.HIGH,
    "grammar_number": ErrorSeverity.MEDIUM,
    "grammar_tense": ErrorSeverity.MEDIUM,
    "grammar_aspect": ErrorSeverity.MEDIUM,
    "grammar_voice": ErrorSeverity.MEDIUM,
    "grammar_pos": ErrorSeverity.MEDIUM,
    "grammar_other": ErrorSeverity.LOW,
    "punctuation": ErrorSeverity.LOW,
    "spelling": ErrorSeverity.MEDIUM,
    "register": ErrorSeverity.LOW,
}


@dataclass
class UAGECAnnotation:
    """Raw annotation from UA-GEC dataset."""

    start: int
    end: int
    source_text: str
    correction: str
    error_type: str
    annotator_id: int = 0


@dataclass
class UAGECDocument:
    """Single document from UA-GEC dataset."""

    doc_id: str
    source_text: str
    target_text: str
    annotations: list[UAGECAnnotation] = field(default_factory=list)


class UAGECLoader:
    """Loader for UA-GEC 2.0 dataset.

    UA-GEC (Ukrainian Grammatical Error Correction) is a corpus
    of Ukrainian texts with annotated grammatical errors.

    Expected directory structure:
    ua_gec/
    ├── train/
    │   ├── source.txt
    │   ├── target.txt
    │   └── annotated.json
    ├── test/
    │   └── ...
    └── README.md
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        """Initialize loader.

        Args:
            data_dir: Path to UA-GEC dataset directory.
        """
        self.data_dir = data_dir

    def load_from_directory(self, path: Path, split: str = "train") -> list[UAGECDocument]:
        """Load documents from UA-GEC directory structure.

        Args:
            path: Path to UA-GEC root directory.
            split: Dataset split (train, test, dev).

        Returns:
            List of parsed documents.
        """
        split_dir = path / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        documents: list[UAGECDocument] = []

        # Try annotated JSON format first
        annotated_file = split_dir / "annotated.json"
        if annotated_file.exists():
            documents = self._load_from_json(annotated_file)
        else:
            # Fall back to parallel corpus format
            source_file = split_dir / "source.txt"
            target_file = split_dir / "target.txt"
            if source_file.exists() and target_file.exists():
                documents = self._load_from_parallel(source_file, target_file)

        return documents

    def _load_from_json(self, json_path: Path) -> list[UAGECDocument]:
        """Load from annotated JSON format."""
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        documents: list[UAGECDocument] = []
        for item in data:
            annotations = [
                UAGECAnnotation(
                    start=ann.get("start", 0),
                    end=ann.get("end", 0),
                    source_text=ann.get("source", ""),
                    correction=ann.get("correction", ""),
                    error_type=ann.get("error_type", ""),
                    annotator_id=ann.get("annotator_id", 0),
                )
                for ann in item.get("annotations", [])
            ]

            doc = UAGECDocument(
                doc_id=item.get("id", ""),
                source_text=item.get("source", ""),
                target_text=item.get("target", ""),
                annotations=annotations,
            )
            documents.append(doc)

        return documents

    def _load_from_parallel(self, source_path: Path, target_path: Path) -> list[UAGECDocument]:
        """Load from parallel source/target files."""
        with open(source_path, encoding="utf-8") as f:
            sources = f.readlines()
        with open(target_path, encoding="utf-8") as f:
            targets = f.readlines()

        if len(sources) != len(targets):
            raise ValueError(f"Source/target mismatch: {len(sources)} vs {len(targets)}")

        documents: list[UAGECDocument] = []
        for i, (source, target) in enumerate(zip(sources, targets, strict=True)):
            doc = UAGECDocument(
                doc_id=f"ua_gec_{i:05d}",
                source_text=source.strip(),
                target_text=target.strip(),
                annotations=[],
            )
            documents.append(doc)

        return documents

    def convert_to_tasks(
        self,
        documents: list[UAGECDocument],
        category_filter: str | None = None,
        max_tasks: int | None = None,
    ) -> list[GECTask]:
        """Convert UA-GEC documents to GECTask schema.

        Args:
            documents: List of UA-GEC documents.
            category_filter: Only include specific category.
            max_tasks: Maximum number of tasks to return.

        Returns:
            List of GEC tasks.
        """
        tasks: list[GECTask] = []

        for doc in documents:
            # Skip if no corrections needed
            if doc.source_text == doc.target_text:
                continue

            # Determine primary category from annotations
            category = self._determine_category(doc)
            if category_filter and category != category_filter:
                continue

            # Convert annotations to error format
            errors = self._convert_annotations(doc)

            task = GECTask(
                id=doc.doc_id,
                category=category,
                input=doc.source_text,
                expected_output=doc.target_text,
                errors=errors,
                source="UA-GEC",
                difficulty=self._determine_difficulty(errors),
            )
            tasks.append(task)

            if max_tasks and len(tasks) >= max_tasks:
                break

        return tasks

    def _determine_category(self, doc: UAGECDocument) -> str:
        """Determine primary error category for document."""
        if not doc.annotations:
            # Infer from text differences
            if self._looks_like_russism(doc.source_text, doc.target_text):
                return "russism"
            return "grammar"

        # Count categories
        category_counts: dict[str, int] = {}
        for ann in doc.annotations:
            mapped = CATEGORY_MAPPING.get(ann.error_type, "grammar")
            category_counts[mapped] = category_counts.get(mapped, 0) + 1

        # Return most common category
        return max(category_counts, key=lambda k: category_counts[k])

    def _looks_like_russism(self, source: str, target: str) -> bool:
        """Heuristic check for common russisms."""
        russism_patterns = [
            (r"прийняти\s+участь", r"взяти\s+участь"),
            (r"міроприємств", r"захід"),
            (r"на\s+протязі", r"протягом"),
            (r"являється", r"є"),
            (r"слідуюч", r"наступн"),
            (r"приймати\s+рішення", r"ухвалювати\s+рішення"),
        ]

        source_lower = source.lower()
        target_lower = target.lower()

        for rus_pattern, ukr_pattern in russism_patterns:
            if re.search(rus_pattern, source_lower) and re.search(ukr_pattern, target_lower):
                return True
        return False

    def _convert_annotations(self, doc: UAGECDocument) -> list[ErrorAnnotation]:
        """Convert UA-GEC annotations to ErrorAnnotation schema."""
        errors: list[ErrorAnnotation] = []

        for ann in doc.annotations:
            mapped_category = CATEGORY_MAPPING.get(ann.error_type, "grammar")
            severity = SEVERITY_MAPPING.get(mapped_category, ErrorSeverity.MEDIUM)

            error = ErrorAnnotation(
                start=ann.start,
                end=ann.end,
                error_type=mapped_category,
                correction=ann.correction,
                severity=severity,
            )
            errors.append(error)

        return errors

    def _determine_difficulty(self, errors: list[ErrorAnnotation]) -> TaskDifficulty:
        """Determine task difficulty based on errors."""
        if not errors:
            return TaskDifficulty.MEDIUM

        # Multiple critical errors = hard
        critical_count = sum(1 for e in errors if e.severity == ErrorSeverity.CRITICAL)
        if critical_count >= 2:
            return TaskDifficulty.HARD

        # Any critical error or many errors = medium
        if critical_count >= 1 or len(errors) >= 3:
            return TaskDifficulty.MEDIUM

        return TaskDifficulty.EASY

    def iter_russism_tasks(
        self, documents: list[UAGECDocument], max_tasks: int | None = None
    ) -> Iterator[GECTask]:
        """Iterate over russism-specific GEC tasks.

        Args:
            documents: List of UA-GEC documents.
            max_tasks: Maximum tasks to yield.

        Yields:
            GEC tasks containing russisms.
        """
        for count, task in enumerate(
            self.convert_to_tasks(documents, category_filter="russism"), start=1
        ):
            yield task
            if max_tasks and count >= max_tasks:
                break

    def get_statistics(self, documents: list[UAGECDocument]) -> dict[str, int]:
        """Get statistics about loaded documents.

        Args:
            documents: List of documents.

        Returns:
            Dictionary with counts by category.
        """
        stats: dict[str, int] = {"total": len(documents)}

        for doc in documents:
            category = self._determine_category(doc)
            stats[category] = stats.get(category, 0) + 1

        return stats
