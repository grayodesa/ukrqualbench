"""Brown-UK corpus loader for validation and false positive testing.

Loads Ukrainian text from the Brown-UK corpus:
https://github.com/lang-uk/brown-uk

License: CC BY 4.0

The Brown-UK corpus contains high-quality Ukrainian texts from
various genres, useful for:
- False positive testing (correct texts shouldn't be flagged)
- Style validation
- Corpus statistics
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

from ukrqualbench.core.schemas import FalsePositiveTask

# Genre mapping for Brown-UK categories
GENRE_MAPPING: dict[str, str] = {
    "press": "journalism",
    "fiction": "fiction",
    "nonfiction": "nonfiction",
    "academic": "academic",
    "official": "official",
    "religious": "religious",
    "miscellaneous": "miscellaneous",
}


@dataclass
class BrownUKDocument:
    """Document from Brown-UK corpus."""

    doc_id: str
    text: str
    genre: str
    source: str | None = None
    author: str | None = None
    year: int | None = None
    sentences: list[str] = field(default_factory=list)


class BrownUKLoader:
    """Loader for Brown-UK corpus.

    The Brown-UK corpus is a balanced corpus of written Ukrainian
    modeled after the Brown corpus of American English.

    Expected formats:
    1. XML format (original)
    2. Plain text files
    3. JSON format
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        """Initialize loader.

        Args:
            data_dir: Path to Brown-UK corpus directory.
        """
        self.data_dir = data_dir

    def load_from_xml(self, xml_path: Path) -> BrownUKDocument:
        """Load single document from XML file.

        Args:
            xml_path: Path to XML file.

        Returns:
            Parsed document.
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Extract metadata
        doc_id = root.get("id", xml_path.stem)
        genre = root.get("genre", "miscellaneous")
        source = root.get("source")
        author = root.get("author")
        year_str = root.get("year")
        year = int(year_str) if year_str else None

        # Extract text
        sentences: list[str] = []
        for sent in root.iter("s"):
            if sent.text:
                sentences.append(sent.text.strip())

        full_text = " ".join(sentences)

        return BrownUKDocument(
            doc_id=doc_id,
            text=full_text,
            genre=GENRE_MAPPING.get(genre, genre),
            source=source,
            author=author,
            year=year,
            sentences=sentences,
        )

    def load_from_directory(self, dir_path: Path) -> list[BrownUKDocument]:
        """Load all documents from directory.

        Args:
            dir_path: Path to directory.

        Returns:
            List of documents.
        """
        documents: list[BrownUKDocument] = []

        # Try XML files first
        for xml_file in dir_path.glob("**/*.xml"):
            try:
                doc = self.load_from_xml(xml_file)
                documents.append(doc)
            except ET.ParseError:
                continue

        # If no XML, try plain text
        if not documents:
            for txt_file in dir_path.glob("**/*.txt"):
                doc = self._load_from_text(txt_file)
                documents.append(doc)

        return documents

    def load_from_json(self, json_path: Path) -> list[BrownUKDocument]:
        """Load from JSON file.

        Args:
            json_path: Path to JSON file.

        Returns:
            List of documents.
        """
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        documents: list[BrownUKDocument] = []
        items = data if isinstance(data, list) else data.get("documents", [])

        for i, item in enumerate(items):
            doc = BrownUKDocument(
                doc_id=item.get("id", f"brown_uk_{i:05d}"),
                text=item.get("text", ""),
                genre=item.get("genre", "miscellaneous"),
                source=item.get("source"),
                author=item.get("author"),
                year=item.get("year"),
                sentences=item.get("sentences", []),
            )
            documents.append(doc)

        return documents

    def _load_from_text(self, txt_path: Path) -> BrownUKDocument:
        """Load from plain text file."""
        with open(txt_path, encoding="utf-8") as f:
            text = f.read()

        # Simple sentence splitting
        sentences = re.split(r"(?<=[.!?])\s+", text)

        return BrownUKDocument(
            doc_id=txt_path.stem,
            text=text,
            genre="miscellaneous",
            sentences=sentences,
        )

    def convert_to_false_positive_tasks(
        self,
        documents: list[BrownUKDocument],
        min_length: int = 50,
        max_length: int = 500,
        max_tasks: int | None = None,
    ) -> list[FalsePositiveTask]:
        """Convert documents to false positive test tasks.

        False positive tasks are texts that should NOT be flagged
        as containing errors - they are correct Ukrainian.

        Args:
            documents: List of documents.
            min_length: Minimum text length in characters.
            max_length: Maximum text length in characters.
            max_tasks: Maximum number of tasks to generate.

        Returns:
            List of false positive tasks.
        """
        tasks: list[FalsePositiveTask] = []

        for doc in documents:
            # Extract suitable passages
            passages = self._extract_passages(doc, min_length, max_length)

            for i, passage in enumerate(passages):
                task = FalsePositiveTask(
                    id=f"{doc.doc_id}_fp_{i:03d}",
                    text=passage,
                    author=doc.author,
                    is_correct=True,
                    notes=f"From {doc.genre} genre, Brown-UK corpus",
                    acceptable_flags=self._get_acceptable_flags(doc.genre),
                )
                tasks.append(task)

                if max_tasks and len(tasks) >= max_tasks:
                    return tasks

        return tasks

    def _extract_passages(
        self, doc: BrownUKDocument, min_length: int, max_length: int
    ) -> list[str]:
        """Extract suitable passages from document."""
        passages: list[str] = []

        if doc.sentences:
            # Build passages from sentences
            current = ""
            for sent in doc.sentences:
                if len(current) + len(sent) + 1 <= max_length:
                    current = f"{current} {sent}".strip()
                else:
                    if len(current) >= min_length:
                        passages.append(current)
                    current = sent

            if len(current) >= min_length:
                passages.append(current)
        else:
            # Split text into chunks
            text = doc.text
            while len(text) >= min_length:
                # Try to split at sentence boundary
                chunk = text[:max_length]
                split_pos = chunk.rfind(".")
                if split_pos < min_length:
                    split_pos = max_length

                passages.append(text[:split_pos + 1].strip())
                text = text[split_pos + 1:].strip()

        return passages

    def _get_acceptable_flags(self, genre: str) -> list[str]:
        """Get acceptable error flags based on genre."""
        # Some genres may have intentional stylistic choices
        acceptable: list[str] = []

        if genre == "fiction":
            acceptable.extend(["archaic_style", "dialectism", "colloquial"])
        elif genre == "academic":
            acceptable.extend(["technical_term", "foreign_word"])
        elif genre == "official":
            acceptable.extend(["bureaucratic_style", "legal_term"])

        return acceptable

    def extract_classic_literature(
        self, documents: list[BrownUKDocument], min_year: int | None = None
    ) -> list[BrownUKDocument]:
        """Extract classic literature documents.

        Args:
            documents: List of all documents.
            min_year: Minimum publication year (for classics).

        Returns:
            Filtered list of classic literature documents.
        """
        classics: list[BrownUKDocument] = []

        for doc in documents:
            if doc.genre != "fiction":
                continue

            # Filter by year if specified
            if min_year and doc.year and doc.year < min_year:
                continue

            # Check for known classic authors
            classic_authors = {
                "шевченко", "франко", "леся українка", "коцюбинський",
                "стефаник", "кобилянська", "тичина", "рильський",
                "малишко", "сосюра", "костенко", "стус",
            }
            if doc.author:
                author_lower = doc.author.lower()
                if any(name in author_lower for name in classic_authors):
                    classics.append(doc)

        return classics

    def iter_by_genre(
        self, documents: list[BrownUKDocument], genre: str
    ) -> Iterator[BrownUKDocument]:
        """Iterate over documents filtered by genre.

        Args:
            documents: List of documents.
            genre: Genre to filter by.

        Yields:
            Documents of specified genre.
        """
        for doc in documents:
            if doc.genre == genre:
                yield doc

    def get_statistics(self, documents: list[BrownUKDocument]) -> dict[str, int]:
        """Get statistics about loaded documents.

        Args:
            documents: List of documents.

        Returns:
            Dictionary with counts by genre.
        """
        stats: dict[str, int] = {
            "total": len(documents),
            "total_sentences": sum(len(d.sentences) for d in documents),
            "total_chars": sum(len(d.text) for d in documents),
        }

        for doc in documents:
            stats[doc.genre] = stats.get(doc.genre, 0) + 1

        return stats
