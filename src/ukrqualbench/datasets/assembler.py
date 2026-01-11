"""Benchmark assembler for building complete benchmark datasets.

Assembles benchmark data from:
1. HuggingFace datasets (ZNO, FLORES, UA-GEC) - preferred
2. External corpus loaders when available locally
3. Embedded synthetic data as fallback
4. JSON task files for Block B
"""

from __future__ import annotations

import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from ukrqualbench.core.schemas import (
    AdversarialTask,
    FalsePositiveTask,
    FreeGenerationTask,
    GECTask,
    LongContextTask,
    MultipleChoiceTask,
    PositiveMarkerTask,
    TaskDifficulty,
    TranslationTask,
)
from ukrqualbench.datasets.loader import (
    BENCHMARK_SPECS,
    BenchmarkData,
    BenchmarkMetadata,
    BlockAData,
    BlockBData,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

BenchmarkVersionType = Literal["lite", "base", "large"]

# Flag to track if datasets library is available
_HF_DATASETS_AVAILABLE: bool | None = None


def _check_hf_datasets() -> bool:
    """Check if HuggingFace datasets library is available."""
    global _HF_DATASETS_AVAILABLE
    if _HF_DATASETS_AVAILABLE is None:
        try:
            import datasets  # noqa: F401

            _HF_DATASETS_AVAILABLE = True
        except ImportError:
            _HF_DATASETS_AVAILABLE = False
            logger.warning(
                "HuggingFace 'datasets' library not installed. "
                "Run 'uv sync' to install it for corpus loading."
            )
    return _HF_DATASETS_AVAILABLE


class BenchmarkAssembler:
    """Assembles complete benchmark datasets from multiple sources.

    Prioritizes external corpus data when available, falls back to
    embedded synthetic data for immediate functionality.
    """

    def __init__(
        self,
        data_dir: Path | None = None,
        seed: int = 42,
        hf_token: str | None = None,
    ) -> None:
        self.data_dir = data_dir or Path("data")
        self.seed = seed
        self.hf_token = hf_token
        self._rng = random.Random(seed)

    def assemble(self, version: BenchmarkVersionType = "base") -> BenchmarkData:
        """Assemble complete benchmark for specified version."""
        spec = BENCHMARK_SPECS[version]

        block_a = self._assemble_block_a(spec["block_a"])
        block_b = self._assemble_block_b(spec["block_b"])

        benchmark = BenchmarkData(version=version, block_a=block_a, block_b=block_b)

        benchmark.metadata = BenchmarkMetadata(
            version=version,
            dataset_hash=benchmark.compute_hash(),
            total_block_a=block_a.total,
            total_block_b=block_b.total,
            created_at=datetime.now().isoformat(),
            sources=self._get_sources_used(),
        )

        return benchmark

    def _assemble_block_a(self, spec: dict[str, int]) -> BlockAData:
        """Assemble Block A tasks."""
        return BlockAData(
            mc_tasks=self._get_mc_tasks(spec["mc"]),
            gec_tasks=self._get_gec_tasks(spec["gec"]),
            translation_tasks=self._get_translation_tasks(spec["translation"]),
            false_positive_tasks=self._get_false_positive_tasks(spec["false_positive"]),
            positive_marker_tasks=self._get_positive_marker_tasks(spec["positive_marker"]),
        )

    def _assemble_block_b(self, spec: dict[str, int]) -> BlockBData:
        """Assemble Block B tasks."""
        return BlockBData(
            generation_tasks=self._get_generation_tasks(spec["free_generation"]),
            adversarial_tasks=self._get_adversarial_tasks(spec["adversarial"]),
            long_context_tasks=self._get_long_context_tasks(spec["long_context"]),
        )

    def _get_mc_tasks(self, count: int) -> list[MultipleChoiceTask]:
        """Get multiple choice tasks, trying external then synthetic."""
        tasks = self._try_load_zno_tasks(count)
        if len(tasks) < count:
            tasks.extend(self._generate_synthetic_mc_tasks(count - len(tasks)))
        return self._sample(tasks, count)

    def _get_gec_tasks(self, count: int) -> list[GECTask]:
        """Get GEC tasks, trying external then synthetic."""
        tasks = self._try_load_uagec_tasks(count)
        if len(tasks) < count:
            tasks.extend(self._generate_synthetic_gec_tasks(count - len(tasks)))
        return self._sample(tasks, count)

    def _get_translation_tasks(self, count: int) -> list[TranslationTask]:
        """Get translation tasks, trying external then synthetic."""
        tasks = self._try_load_flores_tasks(count)
        if len(tasks) < count:
            tasks.extend(self._generate_synthetic_translation_tasks(count - len(tasks)))
        return self._sample(tasks, count)

    def _get_false_positive_tasks(self, count: int) -> list[FalsePositiveTask]:
        """Get false positive tasks from Brown-UK or synthetic."""
        tasks = self._try_load_brownuk_tasks(count)
        if len(tasks) < count:
            tasks.extend(self._generate_synthetic_false_positive_tasks(count - len(tasks)))
        return self._sample(tasks, count)

    def _get_positive_marker_tasks(self, count: int) -> list[PositiveMarkerTask]:
        """Get positive marker tasks (always synthetic)."""
        return self._generate_synthetic_positive_marker_tasks(count)

    def _get_generation_tasks(self, count: int) -> list[FreeGenerationTask]:
        """Get generation tasks from JSON file."""
        tasks = self._load_json_tasks("generation_tasks.json", FreeGenerationTask)
        if len(tasks) < count:
            tasks.extend(self._generate_synthetic_generation_tasks(count - len(tasks)))
        return self._sample(tasks, count)

    def _get_adversarial_tasks(self, count: int) -> list[AdversarialTask]:
        """Get adversarial tasks from JSON file."""
        tasks = self._load_json_tasks("adversarial_tasks.json", AdversarialTask)
        if len(tasks) < count:
            tasks.extend(self._generate_synthetic_adversarial_tasks(count - len(tasks)))
        return self._sample(tasks, count)

    def _get_long_context_tasks(self, count: int) -> list[LongContextTask]:
        """Get long context tasks from JSON or synthetic."""
        tasks = self._load_json_tasks("long_context_tasks.json", LongContextTask)
        if len(tasks) < count:
            tasks.extend(self._generate_synthetic_long_context_tasks(count - len(tasks)))
        return self._sample(tasks, count)

    def _try_load_zno_tasks(self, count: int) -> list[MultipleChoiceTask]:
        """Try to load ZNO tasks from HuggingFace or external corpus."""
        # Try HuggingFace first
        tasks = self._try_load_zno_from_hf(count)
        if tasks:
            return tasks

        # Fall back to local files
        zno_dir = self.data_dir / "external" / "zno"
        if not zno_dir.exists():
            return []

        from ukrqualbench.datasets.sources import ZNOLoader

        loader = ZNOLoader(zno_dir)
        try:
            for json_file in zno_dir.glob("*.json"):
                questions = loader.load_from_json(json_file)
                return loader.convert_to_tasks(questions, max_tasks=count)
        except Exception:
            pass
        return []

    def _try_load_zno_from_hf(self, count: int) -> list[MultipleChoiceTask]:
        """Load ZNO tasks from HuggingFace datasets."""
        if not _check_hf_datasets():
            return []

        try:
            from datasets import load_dataset

            dataset = load_dataset("INSAIT-Institute/zno_ukr", split="test", token=self.hf_token)

            tasks: list[MultipleChoiceTask] = []
            # Sample deterministically
            indices = list(range(len(dataset)))
            self._rng.shuffle(indices)

            for idx in indices[:count]:
                item = dataset[idx]
                question = item["question"]
                answers = item["answers"]
                correct_idx = item["correct_answer"]
                source = item.get("source", "zno")

                # Format options as A), B), C), D)
                formatted_options = [f"{chr(ord('A') + i)}) {opt}" for i, opt in enumerate(answers)]
                correct_letter = chr(ord("A") + correct_idx)

                # Map source to category
                category_map = {
                    "geography": "geography",
                    "history": "history",
                    "ukrainian": "grammar",
                    "math": "logic",
                }
                category = category_map.get(source, "general")

                tasks.append(
                    MultipleChoiceTask(
                        id=f"zno_hf_{idx:05d}",
                        category=category,
                        prompt=question,
                        options=formatted_options,
                        correct=correct_letter,
                        explanation="",
                        difficulty=TaskDifficulty.MEDIUM,
                        source="INSAIT-Institute/zno_ukr",
                    )
                )

            logger.info(f"Loaded {len(tasks)} ZNO tasks from HuggingFace")
            return tasks

        except Exception as e:
            logger.warning(f"Failed to load ZNO from HuggingFace: {e}")
            return []

    def _try_load_uagec_tasks(self, count: int) -> list[GECTask]:
        """Try to load UA-GEC tasks from HuggingFace or external corpus."""
        # Try HuggingFace first
        tasks = self._try_load_uagec_from_hf(count)
        if tasks:
            return tasks

        # Fall back to local files
        uagec_dir = self.data_dir / "external" / "ua-gec"
        if not uagec_dir.exists():
            return []

        from ukrqualbench.datasets.sources import UAGECLoader

        loader = UAGECLoader(uagec_dir)
        try:
            docs = loader.load_from_directory(uagec_dir, split="train")
            return loader.convert_to_tasks(docs, max_tasks=count)
        except Exception:
            pass
        return []

    def _try_load_uagec_from_hf(self, count: int) -> list[GECTask]:
        """Load UA-GEC tasks from HuggingFace datasets."""
        if not _check_hf_datasets():
            return []

        try:
            from datasets import load_dataset

            dataset = load_dataset(
                "osyvokon/ua_gec_instruction_tuning", split="train", token=self.hf_token
            )

            tasks: list[GECTask] = []
            # Sample deterministically
            indices = list(range(len(dataset)))
            self._rng.shuffle(indices)

            for idx in indices[:count]:
                item = dataset[idx]
                # The dataset has instruction/response pairs
                # Extract source (with errors) and target (corrected)
                source_text = item.get("source", item.get("input", ""))
                target_text = item.get("target", item.get("output", ""))

                if not source_text or not target_text or source_text == target_text:
                    continue

                tasks.append(
                    GECTask(
                        id=f"uagec_hf_{idx:05d}",
                        category="grammar",
                        input=source_text,
                        expected_output=target_text,
                        errors=[],
                        source="osyvokon/ua_gec_instruction_tuning",
                        difficulty=TaskDifficulty.MEDIUM,
                    )
                )

                if len(tasks) >= count:
                    break

            logger.info(f"Loaded {len(tasks)} UA-GEC tasks from HuggingFace")
            return tasks

        except Exception as e:
            logger.warning(f"Failed to load UA-GEC from HuggingFace: {e}")
            return []

    def _try_load_flores_tasks(self, count: int) -> list[TranslationTask]:
        """Try to load FLORES tasks from HuggingFace or external corpus."""
        # Try HuggingFace first
        tasks = self._try_load_flores_from_hf(count)
        if tasks:
            return tasks

        # Fall back to local files
        flores_dir = self.data_dir / "external" / "flores"
        if not flores_dir.exists():
            return []

        from ukrqualbench.datasets.sources import FLORESLoader

        loader = FLORESLoader(flores_dir)
        try:
            for json_file in flores_dir.glob("*.json"):
                parallels = loader.load_from_json(json_file, "en", "uk")
                return loader.convert_to_tasks(parallels, max_tasks=count)
        except Exception:
            pass
        return []

    def _try_load_flores_from_hf(self, count: int) -> list[TranslationTask]:
        """Load FLORES tasks from HuggingFace datasets."""
        if not _check_hf_datasets():
            return []

        try:
            from datasets import load_dataset

            tasks: list[TranslationTask] = []

            en_count = int(count * 0.6)
            ru_count = count - en_count

            try:
                en_uk = load_dataset(
                    "facebook/flores",
                    "eng_Latn-ukr_Cyrl",
                    split="devtest",
                    token=self.hf_token,
                )
                indices = list(range(len(en_uk)))
                self._rng.shuffle(indices)

                for idx in indices[:en_count]:
                    item = en_uk[idx]
                    tasks.append(
                        TranslationTask(
                            id=f"flores_en_hf_{idx:05d}",
                            source_lang="en",
                            source=item["sentence_eng_Latn"],
                            reference=item["sentence_ukr_Cyrl"],
                            traps=[],
                            trap_type="none",
                        )
                    )
            except Exception as e:
                logger.warning(f"Failed to load EN-UK FLORES: {e}")

            try:
                ru_uk = load_dataset(
                    "facebook/flores",
                    "rus_Cyrl-ukr_Cyrl",
                    split="devtest",
                    token=self.hf_token,
                )
                indices = list(range(len(ru_uk)))
                self._rng.shuffle(indices)

                for idx in indices[:ru_count]:
                    item = ru_uk[idx]
                    tasks.append(
                        TranslationTask(
                            id=f"flores_ru_hf_{idx:05d}",
                            source_lang="ru",
                            source=item["sentence_rus_Cyrl"],
                            reference=item["sentence_ukr_Cyrl"],
                            traps=[],
                            trap_type="russism",
                        )
                    )
            except Exception as e:
                logger.warning(f"Failed to load RU-UK FLORES: {e}")

            logger.info(f"Loaded {len(tasks)} FLORES tasks from HuggingFace")
            return tasks

        except Exception as e:
            logger.warning(f"Failed to load FLORES from HuggingFace: {e}")
            return []

    def _try_load_brownuk_tasks(self, count: int) -> list[FalsePositiveTask]:
        """Try to load Brown-UK tasks from external corpus."""
        brownuk_dir = self.data_dir / "external" / "brown-uk"
        if not brownuk_dir.exists():
            return []

        from ukrqualbench.datasets.sources import BrownUKLoader

        loader = BrownUKLoader(brownuk_dir)
        try:
            docs = loader.load_from_directory(brownuk_dir)
            return loader.convert_to_false_positive_tasks(docs, max_tasks=count)
        except Exception:
            pass
        return []

    def _load_json_tasks(self, filename: str, model_class: type) -> list:  # type: ignore[type-arg]
        """Load tasks from JSON file in benchmarks directory."""
        path = self.data_dir / "benchmarks" / filename
        if not path.exists():
            return []
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            items = data if isinstance(data, list) else data.get("tasks", [])
            return [model_class.model_validate(item) for item in items]  # type: ignore[attr-defined]
        except Exception:
            return []

    def _sample(self, items: list, count: int) -> list:  # type: ignore[type-arg]
        """Sample items with fixed seed for reproducibility."""
        if len(items) <= count:
            return items
        return self._rng.sample(items, count)

    def _get_sources_used(self) -> list[str]:
        """Get list of data sources used."""
        sources = ["synthetic"]

        # Check HuggingFace availability
        if _check_hf_datasets():
            sources.append("HuggingFace")

        # Check local external directories
        external_dir = self.data_dir / "external"
        if external_dir.exists():
            if (external_dir / "zno").exists():
                sources.append("ZNO-local")
            if (external_dir / "ua-gec").exists():
                sources.append("UA-GEC-local")
            if (external_dir / "flores").exists():
                sources.append("FLORES-local")
            if (external_dir / "brown-uk").exists():
                sources.append("Brown-UK-local")
        return sources

    def _generate_synthetic_mc_tasks(self, count: int) -> list[MultipleChoiceTask]:
        """Generate synthetic multiple choice tasks for Ukrainian grammar."""
        templates = [
            {
                "category": "orthography",
                "prompt": "Виберіть правильний варіант написання:",
                "cases": [
                    (["пів'яблука", "півяблука", "пів яблука"], "A", "Апостроф після пів- перед я"),
                    (
                        ["пів-України", "півУкраїни", "пів України"],
                        "A",
                        "Дефіс у пів- з великою літерою",
                    ),
                    (["п'ятсот", "пятсот", "п'ять сот"], "A", "Апостроф у числівниках"),
                    (["восени", "в осені", "во сені"], "A", "Злите написання прислівників"),
                    (
                        ["будь-який", "будьякий", "будь який"],
                        "A",
                        "Дефіс у неозначених займенниках",
                    ),
                ],
            },
            {
                "category": "punctuation",
                "prompt": "Виберіть речення з правильною пунктуацією:",
                "cases": [
                    (
                        [
                            "Він прийшов, але нікого не застав.",
                            "Він прийшов але нікого не застав.",
                            "Він прийшов, але, нікого не застав.",
                        ],
                        "A",
                        "Кома перед але",
                    ),
                    (
                        [
                            "Мама сказала, що прийде пізніше.",
                            "Мама сказала що прийде пізніше.",
                            "Мама, сказала що прийде пізніше.",
                        ],
                        "A",
                        "Кома перед що",
                    ),
                ],
            },
            {
                "category": "grammar_case",
                "prompt": "Виберіть правильну форму слова:",
                "cases": [
                    (["батькові", "батьку", "батька"], "A", "Давальний відмінок з -ові"),
                    (["Києві", "Києву", "Київі"], "A", "Місцевий відмінок -і"),
                    (["брату", "братові", "братом"], "B", "Давальний з -ові для осіб"),
                    (["сину", "синові", "сином"], "B", "Давальний з -ові для осіб"),
                ],
            },
            {
                "category": "russism_detection",
                "prompt": "Виберіть літературну українську форму:",
                "cases": [
                    (
                        ["взяти участь", "прийняти участь", "приймати участь"],
                        "A",
                        "Русизм 'прийняти участь'",
                    ),
                    (["захід", "міроприємство", "мероприятіє"], "A", "Русизм 'міроприємство'"),
                    (["протягом", "на протязі", "в продовж"], "A", "Русизм 'на протязі'"),
                    (["є", "являється", "являеться"], "A", "Русизм 'являється'"),
                    (["наступний", "слідуючий", "слідующий"], "A", "Русизм 'слідуючий'"),
                    (
                        ["вжити заходів", "прийняти міри", "приняти меры"],
                        "A",
                        "Русизм 'прийняти міри'",
                    ),
                    (
                        ["ухвалити рішення", "прийняти рішення", "приняти решення"],
                        "A",
                        "Русизм 'прийняти рішення'",
                    ),
                    (
                        ["здобути досвід", "отримати досвід", "получити досвід"],
                        "A",
                        "Русизм 'отримати досвід'",
                    ),
                ],
            },
            {
                "category": "vocative",
                "prompt": "Виберіть правильну кличну форму:",
                "cases": [
                    (["Андрію", "Андрій", "Андріє"], "A", "Кличний від Андрій"),
                    (["Сергію", "Сергій", "Сергіє"], "A", "Кличний від Сергій"),
                    (["Маріє", "Марія", "Марійо"], "A", "Кличний від Марія"),
                    (["Олено", "Олена", "Оленко"], "A", "Кличний від Олена"),
                    (["друже", "друг", "другу"], "A", "Кличний від друг"),
                ],
            },
        ]

        tasks: list[MultipleChoiceTask] = []
        task_id = 0

        while len(tasks) < count:
            for template in templates:
                for options, correct, explanation in template["cases"]:
                    if len(tasks) >= count:
                        break
                    formatted_options = [
                        f"{chr(ord('A') + i)}) {opt}" for i, opt in enumerate(options)
                    ]
                    tasks.append(
                        MultipleChoiceTask(
                            id=f"mc_synth_{task_id:04d}",
                            category=str(template["category"]),
                            prompt=str(template["prompt"]),
                            options=formatted_options,
                            correct=str(correct),
                            explanation=str(explanation),
                            difficulty=TaskDifficulty.MEDIUM,
                            source="synthetic",
                        )
                    )
                    task_id += 1

        return tasks[:count]

    def _generate_synthetic_gec_tasks(self, count: int) -> list[GECTask]:
        """Generate synthetic GEC tasks with russisms and errors."""
        error_pairs = [
            ("Треба прийняти участь у заході.", "Треба взяти участь у заході.", "russism"),
            ("На протязі року ми працювали.", "Протягом року ми працювали.", "russism"),
            ("Це являється головною проблемою.", "Це є головною проблемою.", "russism"),
            ("Слідуючий крок буде важливим.", "Наступний крок буде важливим.", "russism"),
            ("Треба прийняти міри.", "Треба вжити заходів.", "russism"),
            ("Збори будуть мати місце завтра.", "Збори відбудуться завтра.", "russism"),
            ("Прийняти рішення до кінця дня.", "Ухвалити рішення до кінця дня.", "russism"),
            ("Отримати досвід роботи.", "Здобути досвід роботи.", "russism"),
            ("В даному випадку це важливо.", "У цьому випадку це важливо.", "russism"),
            ("Дякуючи вашій допомозі.", "Завдяки вашій допомозі.", "russism"),
            ("Не дивлячись на це.", "Попри це.", "russism"),
            ("Самий кращий варіант.", "Найкращий варіант.", "russism"),
            ("По відношенню до цього питання.", "Щодо цього питання.", "russism"),
            ("Приводжу до вашого відома.", "Повідомляю вам.", "russism"),
            ("В кінці кінців.", "Врешті-решт.", "russism"),
        ]

        tasks: list[GECTask] = []
        for i, (input_text, expected, category) in enumerate(error_pairs[:count]):
            tasks.append(
                GECTask(
                    id=f"gec_synth_{i:04d}",
                    category=category,
                    input=input_text,
                    expected_output=expected,
                    errors=[],
                    source="synthetic",
                    difficulty=TaskDifficulty.MEDIUM,
                )
            )

        while len(tasks) < count:
            idx = len(tasks) % len(error_pairs)
            inp, exp, cat = error_pairs[idx]
            tasks.append(
                GECTask(
                    id=f"gec_synth_{len(tasks):04d}",
                    category=cat,
                    input=inp,
                    expected_output=exp,
                    errors=[],
                    source="synthetic",
                    difficulty=TaskDifficulty.MEDIUM,
                )
            )

        return tasks[:count]

    def _generate_synthetic_translation_tasks(self, count: int) -> list[TranslationTask]:
        """Generate synthetic translation tasks with traps."""
        ru_uk_pairs = [
            (
                "Мы должны принять участие в мероприятии.",
                "Ми маємо взяти участь у заході.",
                ["прийняти участь", "міроприємство"],
            ),
            ("На протяжении всего дня шел дождь.", "Протягом усього дня йшов дощ.", ["на протязі"]),
            ("Это является основной причиной.", "Це є основною причиною.", ["являється"]),
            (
                "Следующий вопрос касается экономики.",
                "Наступне питання стосується економіки.",
                ["слідуючий"],
            ),
            ("Необходимо принять меры.", "Необхідно вжити заходів.", ["прийняти міри"]),
            ("Собрание будет иметь место завтра.", "Збори відбудуться завтра.", ["мати місце"]),
            (
                "Принять решение до конца дня.",
                "Ухвалити рішення до кінця дня.",
                ["прийняти рішення"],
            ),
            ("Получить опыт работы.", "Здобути досвід роботи.", ["отримати досвід"]),
            ("В данном случае это важно.", "У цьому випадку це важливо.", ["в даному випадку"]),
            ("Благодаря вашей помощи.", "Завдяки вашій допомозі.", ["дякуючи"]),
        ]

        tasks: list[TranslationTask] = []
        for i, (source, reference, traps) in enumerate(ru_uk_pairs[:count]):
            tasks.append(
                TranslationTask(
                    id=f"trans_synth_{i:04d}",
                    source_lang="ru",
                    source=source,
                    reference=reference,
                    traps=traps,
                    trap_type="russism",
                )
            )

        while len(tasks) < count:
            idx = len(tasks) % len(ru_uk_pairs)
            src, ref, trp = ru_uk_pairs[idx]
            tasks.append(
                TranslationTask(
                    id=f"trans_synth_{len(tasks):04d}",
                    source_lang="ru",
                    source=src,
                    reference=ref,
                    traps=trp,
                    trap_type="russism",
                )
            )

        return tasks[:count]

    def _generate_synthetic_false_positive_tasks(self, count: int) -> list[FalsePositiveTask]:
        """Generate false positive tasks from classic Ukrainian literature."""
        classics = [
            (
                "Реве та стогне Дніпр широкий, Сердитий вітер завива, Додолу верби гне високі, Горами хвилю підійма.",
                "Тарас Шевченко",
            ),
            (
                "Як умру, то поховайте Мене на могилі, Серед степу широкого, На Вкраїні милій.",
                "Тарас Шевченко",
            ),
            ("Ні, я жива! Я буду вічно жити! Я в серці маю те, що не вмирає.", "Леся Українка"),
            ("Земле моя, всеплодющая мати, Сили, що творять в твоїй глибині!", "Іван Франко"),
            ("І все-таки до тебе думка лине, Мій занапащений, нещасний краю!", "Василь Стус"),
            ("Любіть Україну, як сонце, любіть, Як вітер, і трави, і води.", "Володимир Сосюра"),
            ("Чого являєшся мені у сні? Чого звертаєш ти до мене очі?", "Ліна Костенко"),
            (
                "Мріють крилами з туману лебеді рожеві, Сиплють ночі у лимани зорі сургучеві.",
                "Павло Тичина",
            ),
            (
                "Ой не ріж косою, тато, Тій травички молодої, Бо то ж рідна Україна Плаче, тужить надо мною.",
                "Богдан-Ігор Антонич",
            ),
            (
                "Душа моя сумує, мов той сірий ранок, Що журиться по листю обсипанім.",
                "Олександр Олесь",
            ),
        ]

        tasks: list[FalsePositiveTask] = []
        for i, (text, author) in enumerate(classics[:count]):
            tasks.append(
                FalsePositiveTask(
                    id=f"fp_synth_{i:04d}",
                    text=text,
                    author=author,
                    is_correct=True,
                    notes="Classic Ukrainian literature",
                    acceptable_flags=["archaic_style"],
                )
            )

        while len(tasks) < count:
            idx = len(tasks) % len(classics)
            txt, auth = classics[idx]
            tasks.append(
                FalsePositiveTask(
                    id=f"fp_synth_{len(tasks):04d}",
                    text=txt,
                    author=auth,
                    is_correct=True,
                    notes="Classic Ukrainian literature",
                    acceptable_flags=["archaic_style"],
                )
            )

        return tasks[:count]

    def _generate_synthetic_positive_marker_tasks(self, count: int) -> list[PositiveMarkerTask]:
        """Generate positive marker detection tasks."""
        markers = [
            (
                "vocative",
                "Звертання до людини",
                "Пане Андрію, допоможіть мені.",
                ["Пан Андрій"],
                r"\b(Пане|Пані|Друже|Брате|Сестро)\s+[А-ЯІЇЄҐ][а-яіїєґ']+[ію]\b",
            ),
            (
                "vocative",
                "Кличний відмінок імені",
                "Маріє, ти чула новину?",
                ["Марія"],
                r"\b[А-ЯІЇЄҐ][а-яіїєґ']+[ею]\b,",
            ),
            (
                "particle_bo",
                "Частка 'бо'",
                "Він не прийшов, бо був зайнятий.",
                ["тому що"],
                r"\bбо\b",
            ),
            ("particle_zh", "Частка 'ж/же'", "Це ж очевидно!", ["ведь"], r"\b(ж|же)\b"),
            ("particle_khiba", "Частка 'хіба'", "Хіба це можливо?", ["разве"], r"\bхіба\b"),
            ("particle_nevzhe", "Частка 'невже'", "Невже ти не знаєш?", ["неужели"], r"\bневже\b"),
            (
                "conjunction_prote",
                "Сполучник 'проте'",
                "Було холодно, проте ми вийшли.",
                ["однако"],
                r"\bпроте\b",
            ),
            (
                "conjunction_otzhe",
                "Сполучник 'отже'",
                "Отже, рішення прийнято.",
                ["значит"],
                r"\bотже\b",
            ),
            (
                "conjunction_utim",
                "Сполучник 'утім'",
                "Утім, це не має значення.",
                ["впрочем"],
                r"\bутім\b",
            ),
            (
                "conjunction_zato",
                "Сполучник 'зате'",
                "Він не прийшов, зате подзвонив.",
                ["зато"],
                r"\bзате\b",
            ),
        ]

        tasks: list[PositiveMarkerTask] = []
        for i, (category, context, native, non_native, regex) in enumerate(markers[:count]):
            tasks.append(
                PositiveMarkerTask(
                    id=f"pm_synth_{i:04d}",
                    category=category,
                    context=context,
                    native_form=native,
                    non_native_forms=non_native,
                    marker_regex=regex,
                )
            )

        while len(tasks) < count:
            idx = len(tasks) % len(markers)
            cat, ctx, nat, nn, rgx = markers[idx]
            tasks.append(
                PositiveMarkerTask(
                    id=f"pm_synth_{len(tasks):04d}",
                    category=cat,
                    context=ctx,
                    native_form=nat,
                    non_native_forms=nn,
                    marker_regex=rgx,
                )
            )

        return tasks[:count]

    def _generate_synthetic_generation_tasks(self, count: int) -> list[FreeGenerationTask]:
        """Generate additional generation tasks if JSON has too few."""
        templates = [
            ("explanation", "Поясни, що таке {topic}.", 50, 300),
            ("advice", "Дай поради щодо {topic}.", 60, 300),
            ("creative", "Напиши короткий текст про {topic}.", 80, 400),
            ("technical", "Опиши процес {topic}.", 80, 350),
            ("everyday", "Розкажи про {topic}.", 60, 300),
            ("comparison", "Порівняй {topic} з іншими сферами.", 70, 350),
            ("history", "Розкажи про історію {topic}.", 80, 400),
            ("future", "Як ти бачиш майбутнє {topic}?", 70, 350),
        ]

        topics = [
            "інтернет",
            "екологія",
            "здоров'я",
            "освіта",
            "технології",
            "культура",
            "спорт",
            "мистецтво",
            "наука",
            "подорожі",
            "економіка",
            "політика",
            "музика",
            "кіно",
            "література",
        ]

        tasks: list[FreeGenerationTask] = []
        task_id = 100

        while len(tasks) < count:
            for topic in topics:
                for category, prompt_template, min_tok, max_tok in templates:
                    if len(tasks) >= count:
                        break
                    tasks.append(
                        FreeGenerationTask(
                            id=f"gen_synth_{task_id:04d}",
                            category=category,
                            prompt=prompt_template.format(topic=topic),
                            min_tokens=min_tok,
                            max_tokens=max_tok,
                        )
                    )
                    task_id += 1
                if len(tasks) >= count:
                    break

        return tasks[:count]

    def _generate_synthetic_adversarial_tasks(self, count: int) -> list[AdversarialTask]:
        """Generate additional adversarial tasks if JSON has too few."""
        bad_phrases = [
            ("прийняти участь в міроприємстві", ["прийняти участь", "міроприємство"]),
            ("на протязі всього часу", ["на протязі"]),
            ("являється головним фактором", ["являється"]),
            ("слідуючим чином", ["слідуючий"]),
            ("приймати міри", ["приймати міри"]),
            ("самий кращий варіант", ["самий кращий"]),
            ("дякуючи вашій допомозі", ["дякуючи"]),
            ("по відношенню до цього", ["по відношенню"]),
            ("в даному випадку", ["в даному випадку"]),
            ("отримати досвід роботи", ["отримати досвід"]),
            ("прийняти рішення негайно", ["прийняти рішення"]),
            ("не дивлячись на складнощі", ["не дивлячись"]),
            ("збори будуть мати місце", ["мати місце"]),
            ("в кінці кінців вирішили", ["в кінці кінців"]),
            ("приводжу до вашого відома", ["приводжу до відома"]),
            ("любий ціною досягнути мети", ["любий ціною"]),
            ("вибачаюсь за незручності", ["вибачаюсь"]),
            ("представляю вашій увазі", ["представляю"]),
            ("приймати до уваги факти", ["приймати до уваги"]),
            ("відноситись до справи серйозно", ["відноситись"]),
        ]

        prompt_templates = [
            "Користувач написав: '{bad}'. Допоможи йому правильно сформулювати думку.",
            "Виправ помилки в тексті: '{bad}'",
            "Перефразуй цей вираз літературною мовою: '{bad}'",
        ]

        tasks: list[AdversarialTask] = []
        task_id = 100

        while len(tasks) < count:
            for bad, traps in bad_phrases:
                for template in prompt_templates:
                    if len(tasks) >= count:
                        break
                    tasks.append(
                        AdversarialTask(
                            id=f"adv_synth_{task_id:04d}",
                            category="russism_trap",
                            prompt=template.format(bad=bad),
                            traps_in_prompt=traps,
                            instruction="Модель НЕ повинна копіювати русизми",
                        )
                    )
                    task_id += 1
                if len(tasks) >= count:
                    break

        return tasks[:count]

    def _generate_synthetic_long_context_tasks(self, count: int) -> list[LongContextTask]:
        """Generate long context tasks for testing language degradation."""
        tasks: list[LongContextTask] = []

        base_messages = [
            {"role": "user", "content": "Привіт! Розкажи мені про історію України."},
            {
                "role": "assistant",
                "content": "Вітаю! Історія України багата та цікава. Вона охоплює тисячоліття...",
            },
            {"role": "user", "content": "А що було в часи Київської Русі?"},
            {
                "role": "assistant",
                "content": "Київська Русь була потужною середньовічною державою...",
            },
            {"role": "user", "content": "Продовж, будь ласка."},
            {"role": "assistant", "content": "Після занепаду Київської Русі український народ..."},
        ]

        for i in range(count):
            extended_messages = base_messages.copy()
            for j in range(10):
                extended_messages.append(
                    {"role": "user", "content": f"Розкажи більше про подію {j + 1}."}
                )
                extended_messages.append(
                    {"role": "assistant", "content": "Ця подія була важливою для розвитку..."}
                )

            tasks.append(
                LongContextTask(
                    id=f"lc_synth_{i:04d}",
                    category="consistency",
                    messages=extended_messages,
                    total_tokens=2000 + i * 500,
                    checkpoints=[0.25, 0.5, 0.75, 1.0],
                    metrics=["russism_rate", "fertility", "vocative_usage"],
                )
            )

        return tasks[:count]


def create_benchmark_assembler(
    data_dir: Path | None = None,
    seed: int = 42,
    hf_token: str | None = None,
) -> BenchmarkAssembler:
    """Factory function to create benchmark assembler."""
    return BenchmarkAssembler(data_dir=data_dir, seed=seed, hf_token=hf_token)
