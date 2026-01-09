"""FLORES-200 dataset loader for translation tasks.

Loads parallel translation data from the FLORES-200 dataset:
https://github.com/facebookresearch/flores

License: CC BY-SA 4.0
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict

if TYPE_CHECKING:
    from collections.abc import Iterator

from ukrqualbench.core.schemas import TranslationTask


class TrapSentence(TypedDict):
    """Type for trap sentence dictionary."""

    ru: str
    uk: str
    traps: list[str]


# Language code mapping
LANG_CODE_MAPPING: dict[str, str] = {
    "eng_Latn": "en",
    "rus_Cyrl": "ru",
    "ukr_Cyrl": "uk",
    "english": "en",
    "russian": "ru",
    "ukrainian": "uk",
    "en": "en",
    "ru": "ru",
    "uk": "uk",
}

# Common russism traps in Russian→Ukrainian translation
RU_UK_TRAPS: dict[str, list[str]] = {
    "принять участие": ["прийняти участь"],
    "мероприятие": ["міроприємство"],
    "на протяжении": ["на протязі"],
    "является": ["являється"],
    "следующий": ["слідуючий"],
    "получить опыт": ["отримати досвід"],
    "принять решение": ["прийняти рішення"],
    "предпринять меры": ["прийняти міри"],
    "иметь место": ["мати місце"],
    "в значительной мере": ["в значній мірі"],
}

# Common anglicism traps in English→Ukrainian translation
EN_UK_TRAPS: dict[str, list[str]] = {
    "make sense": ["мати сенс"],  # Should be "мати рацію" or context-dependent
    "take place": ["брати місце"],  # Should be "відбуватися"
    "by the way": ["до речі"],  # OK but sometimes overused
    "in fact": ["фактично"],  # OK but sometimes overused
}


@dataclass
class FLORESParallel:
    """Parallel sentence from FLORES dataset."""

    sentence_id: str
    source_text: str
    source_lang: str
    target_text: str
    target_lang: str
    domain: str | None = None


class FLORESLoader:
    """Loader for FLORES-200 translation dataset.

    FLORES-200 is a many-to-many multilingual translation benchmark
    with high-quality human translations for 200 languages.

    Expected formats:
    1. Parallel text files (source.txt, target.txt)
    2. JSON format with aligned sentences
    3. TSV format with tab-separated columns
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        """Initialize loader.

        Args:
            data_dir: Path to FLORES dataset directory.
        """
        self.data_dir = data_dir

    def load_parallel_files(
        self,
        source_path: Path,
        target_path: Path,
        source_lang: str,
        target_lang: str,
    ) -> list[FLORESParallel]:
        """Load from parallel text files.

        Args:
            source_path: Path to source language file.
            target_path: Path to target language file.
            source_lang: Source language code.
            target_lang: Target language code.

        Returns:
            List of parallel sentences.
        """
        with open(source_path, encoding="utf-8") as f:
            sources = f.readlines()
        with open(target_path, encoding="utf-8") as f:
            targets = f.readlines()

        if len(sources) != len(targets):
            raise ValueError(f"Parallel files mismatch: {len(sources)} vs {len(targets)}")

        # Normalize language codes
        src_lang = LANG_CODE_MAPPING.get(source_lang, source_lang)
        tgt_lang = LANG_CODE_MAPPING.get(target_lang, target_lang)

        parallels: list[FLORESParallel] = []
        for i, (src, tgt) in enumerate(zip(sources, targets, strict=True)):
            parallel = FLORESParallel(
                sentence_id=f"flores_{src_lang}_{tgt_lang}_{i:05d}",
                source_text=src.strip(),
                source_lang=src_lang,
                target_text=tgt.strip(),
                target_lang=tgt_lang,
            )
            parallels.append(parallel)

        return parallels

    def load_from_json(
        self, json_path: Path, source_lang: str, target_lang: str
    ) -> list[FLORESParallel]:
        """Load from JSON file.

        Args:
            json_path: Path to JSON file.
            source_lang: Source language code.
            target_lang: Target language code.

        Returns:
            List of parallel sentences.
        """
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        src_lang = LANG_CODE_MAPPING.get(source_lang, source_lang)
        tgt_lang = LANG_CODE_MAPPING.get(target_lang, target_lang)

        parallels: list[FLORESParallel] = []
        items = data if isinstance(data, list) else data.get("data", [])

        for i, item in enumerate(items):
            source = item.get(source_lang) or item.get("source") or item.get("src")
            target = item.get(target_lang) or item.get("target") or item.get("tgt")

            if not source or not target:
                continue

            parallel = FLORESParallel(
                sentence_id=item.get("id", f"flores_{src_lang}_{tgt_lang}_{i:05d}"),
                source_text=source,
                source_lang=src_lang,
                target_text=target,
                target_lang=tgt_lang,
                domain=item.get("domain"),
            )
            parallels.append(parallel)

        return parallels

    def load_from_tsv(
        self,
        tsv_path: Path,
        source_lang: str,
        target_lang: str,
        source_col: int = 0,
        target_col: int = 1,
    ) -> list[FLORESParallel]:
        """Load from TSV file.

        Args:
            tsv_path: Path to TSV file.
            source_lang: Source language code.
            target_lang: Target language code.
            source_col: Column index for source text.
            target_col: Column index for target text.

        Returns:
            List of parallel sentences.
        """
        src_lang = LANG_CODE_MAPPING.get(source_lang, source_lang)
        tgt_lang = LANG_CODE_MAPPING.get(target_lang, target_lang)

        parallels: list[FLORESParallel] = []

        with open(tsv_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue

                parts = line.strip().split("\t")
                if len(parts) <= max(source_col, target_col):
                    continue

                parallel = FLORESParallel(
                    sentence_id=f"flores_{src_lang}_{tgt_lang}_{i:05d}",
                    source_text=parts[source_col],
                    source_lang=src_lang,
                    target_text=parts[target_col],
                    target_lang=tgt_lang,
                )
                parallels.append(parallel)

        return parallels

    def convert_to_tasks(
        self,
        parallels: list[FLORESParallel],
        direction: Literal["en-uk", "ru-uk"] | None = None,
        max_tasks: int | None = None,
    ) -> list[TranslationTask]:
        """Convert parallel sentences to TranslationTask schema.

        Args:
            parallels: List of parallel sentences.
            direction: Translation direction filter.
            max_tasks: Maximum number of tasks to return.

        Returns:
            List of translation tasks.
        """
        tasks: list[TranslationTask] = []

        for p in parallels:
            # Filter by direction if specified
            if direction:
                src, tgt = direction.split("-")
                if p.source_lang != src or p.target_lang != tgt:
                    continue

            # Only accept translations to Ukrainian
            if p.target_lang != "uk":
                continue

            # Only accept from English or Russian
            if p.source_lang not in ("en", "ru"):
                continue

            # Detect potential traps
            traps = self._detect_traps(p)

            task = TranslationTask(
                id=p.sentence_id,
                source_lang=p.source_lang,  # type: ignore[arg-type]
                source=p.source_text,
                reference=p.target_text,
                traps=traps,
                trap_type="russism" if p.source_lang == "ru" else "anglicism",
            )
            tasks.append(task)

            if max_tasks and len(tasks) >= max_tasks:
                break

        return tasks

    def _detect_traps(self, parallel: FLORESParallel) -> list[str]:
        """Detect potential translation traps in source text.

        Args:
            parallel: Parallel sentence.

        Returns:
            List of detected traps.
        """
        traps: list[str] = []
        source_lower = parallel.source_text.lower()

        if parallel.source_lang == "ru":
            trap_dict = RU_UK_TRAPS
        elif parallel.source_lang == "en":
            trap_dict = EN_UK_TRAPS
        else:
            return []

        for source_pattern, ukr_traps in trap_dict.items():
            if source_pattern.lower() in source_lower:
                traps.extend(ukr_traps)

        return traps

    def iter_by_direction(
        self, parallels: list[FLORESParallel], direction: Literal["en-uk", "ru-uk"]
    ) -> Iterator[TranslationTask]:
        """Iterate over tasks filtered by translation direction.

        Args:
            parallels: List of parallel sentences.
            direction: Translation direction.

        Yields:
            Translation tasks.
        """
        yield from self.convert_to_tasks(parallels, direction=direction)

    def get_statistics(self, parallels: list[FLORESParallel]) -> dict[str, int]:
        """Get statistics about loaded parallels.

        Args:
            parallels: List of parallel sentences.

        Returns:
            Dictionary with counts by language pair.
        """
        stats: dict[str, int] = {"total": len(parallels)}

        for p in parallels:
            pair = f"{p.source_lang}-{p.target_lang}"
            stats[pair] = stats.get(pair, 0) + 1

        return stats


def create_ru_uk_trap_tasks(
    num_tasks: int = 100,
) -> list[TranslationTask]:
    """Create synthetic Russian→Ukrainian translation tasks with traps.

    This creates tasks specifically designed to test for common
    russism calques in translation.

    Args:
        num_tasks: Number of tasks to create.

    Returns:
        List of translation tasks with traps.
    """
    # Base sentences with common russisms
    trap_sentences: list[TrapSentence] = [
        {
            "ru": "Мы должны принять участие в этом мероприятии.",
            "uk": "Ми маємо взяти участь у цьому заході.",
            "traps": ["прийняти участь", "міроприємство"],
        },
        {
            "ru": "На протяжении всего дня шел дождь.",
            "uk": "Протягом усього дня йшов дощ.",
            "traps": ["на протязі"],
        },
        {
            "ru": "Это является основной причиной проблемы.",
            "uk": "Це є основною причиною проблеми.",
            "traps": ["являється"],
        },
        {
            "ru": "Следующий вопрос касается экономики.",
            "uk": "Наступне питання стосується економіки.",
            "traps": ["слідуючий"],
        },
        {
            "ru": "Необходимо принять меры для решения проблемы.",
            "uk": "Необхідно вжити заходів для розв'язання проблеми.",
            "traps": ["прийняти міри"],
        },
        {
            "ru": "Они получили большой опыт работы.",
            "uk": "Вони здобули великий досвід роботи.",
            "traps": ["отримати досвід"],
        },
        {
            "ru": "В данном случае мы имеем дело с исключением.",
            "uk": "У цьому випадку ми маємо справу з винятком.",
            "traps": ["в даному випадку"],
        },
        {
            "ru": "Собрание будет иметь место завтра.",
            "uk": "Збори відбудуться завтра.",
            "traps": ["мати місце"],
        },
        {
            "ru": "Прошу вас принять решение как можно скорее.",
            "uk": "Прошу вас ухвалити рішення якомога швидше.",
            "traps": ["прийняти рішення"],
        },
        {
            "ru": "Это в значительной мере зависит от обстоятельств.",
            "uk": "Це значною мірою залежить від обставин.",
            "traps": ["в значній мірі"],
        },
    ]

    tasks: list[TranslationTask] = []
    for i in range(min(num_tasks, len(trap_sentences))):
        sent = trap_sentences[i % len(trap_sentences)]
        task = TranslationTask(
            id=f"trans_ru_{i:03d}",
            source_lang="ru",
            source=sent["ru"],
            reference=sent["uk"],
            traps=sent["traps"],
            trap_type="russism",
        )
        tasks.append(task)

    return tasks
