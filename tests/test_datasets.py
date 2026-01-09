"""Tests for dataset loaders and benchmark assembly."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from ukrqualbench.core.schemas import (
    FreeGenerationTask,
    GECTask,
    MultipleChoiceTask,
    TranslationTask,
)
from ukrqualbench.datasets.loader import (
    BENCHMARK_SPECS,
    BenchmarkData,
    BenchmarkLoader,
    BlockAData,
    BlockBData,
)
from ukrqualbench.datasets.sources.flores import FLORESLoader, create_ru_uk_trap_tasks
from ukrqualbench.datasets.sources.ua_gec import UAGECLoader
from ukrqualbench.datasets.sources.zno import ZNOLoader


class TestBenchmarkSpecs:
    """Tests for benchmark specifications."""

    def test_all_versions_defined(self) -> None:
        """Test that lite, base, large versions are defined."""
        assert "lite" in BENCHMARK_SPECS
        assert "base" in BENCHMARK_SPECS
        assert "large" in BENCHMARK_SPECS

    def test_lite_smaller_than_base(self) -> None:
        """Test that lite is smaller than base."""
        lite = BENCHMARK_SPECS["lite"]
        base = BENCHMARK_SPECS["base"]

        assert lite["block_a"]["mc"] < base["block_a"]["mc"]
        assert lite["block_b"]["free_generation"] < base["block_b"]["free_generation"]

    def test_base_smaller_than_large(self) -> None:
        """Test that base is smaller than large."""
        base = BENCHMARK_SPECS["base"]
        large = BENCHMARK_SPECS["large"]

        assert base["block_a"]["mc"] < large["block_a"]["mc"]
        assert base["block_b"]["free_generation"] < large["block_b"]["free_generation"]

    def test_all_task_types_in_specs(self) -> None:
        """Test that all task types are specified."""
        for version in BENCHMARK_SPECS.values():
            # Block A
            assert "mc" in version["block_a"]
            assert "gec" in version["block_a"]
            assert "translation" in version["block_a"]
            assert "false_positive" in version["block_a"]
            assert "positive_marker" in version["block_a"]

            # Block B
            assert "free_generation" in version["block_b"]
            assert "adversarial" in version["block_b"]
            assert "long_context" in version["block_b"]


class TestBlockAData:
    """Tests for BlockAData container."""

    def test_empty_data(self) -> None:
        """Test empty BlockAData."""
        data = BlockAData()
        assert data.total == 0
        assert len(data.all_tasks()) == 0

    def test_total_count(self) -> None:
        """Test total count calculation."""
        data = BlockAData(
            mc_tasks=[
                MultipleChoiceTask(
                    id="mc_1",
                    category="test",
                    prompt="Question?",
                    options=["A) Yes", "B) No"],
                    correct="A",
                    source="test",
                )
            ],
            gec_tasks=[
                GECTask(
                    id="gec_1",
                    category="test",
                    input="error text",
                    expected_output="correct text",
                )
            ],
        )
        assert data.total == 2

    def test_all_tasks_returns_flat_list(self) -> None:
        """Test that all_tasks returns a flat list."""
        data = BlockAData(
            mc_tasks=[
                MultipleChoiceTask(
                    id="mc_1",
                    category="test",
                    prompt="Q?",
                    options=["A", "B"],
                    correct="A",
                    source="test",
                )
            ],
            translation_tasks=[
                TranslationTask(
                    id="trans_1",
                    source_lang="en",
                    source="Hello",
                    reference="Привіт",
                )
            ],
        )
        tasks = data.all_tasks()
        assert len(tasks) == 2
        assert tasks[0].id == "mc_1"
        assert tasks[1].id == "trans_1"


class TestBlockBData:
    """Tests for BlockBData container."""

    def test_empty_data(self) -> None:
        """Test empty BlockBData."""
        data = BlockBData()
        assert data.total == 0
        assert len(data.all_tasks()) == 0

    def test_total_count(self) -> None:
        """Test total count calculation."""
        data = BlockBData(
            generation_tasks=[
                FreeGenerationTask(
                    id="gen_1",
                    category="explanation",
                    prompt="Explain something.",
                )
            ],
        )
        assert data.total == 1


class TestBenchmarkData:
    """Tests for BenchmarkData."""

    def test_total_tasks(self) -> None:
        """Test total task count."""
        block_a = BlockAData(
            mc_tasks=[
                MultipleChoiceTask(
                    id="mc_1",
                    category="test",
                    prompt="Q?",
                    options=["A", "B"],
                    correct="A",
                    source="test",
                )
            ]
        )
        block_b = BlockBData(
            generation_tasks=[
                FreeGenerationTask(
                    id="gen_1",
                    category="test",
                    prompt="Generate.",
                )
            ]
        )
        data = BenchmarkData(version="lite", block_a=block_a, block_b=block_b)
        assert data.total_tasks == 2

    def test_compute_hash_deterministic(self) -> None:
        """Test that hash is deterministic."""
        block_a = BlockAData(
            mc_tasks=[
                MultipleChoiceTask(
                    id="mc_1",
                    category="test",
                    prompt="Q?",
                    options=["A", "B"],
                    correct="A",
                    source="test",
                )
            ]
        )
        block_b = BlockBData()
        data1 = BenchmarkData(version="lite", block_a=block_a, block_b=block_b)
        data2 = BenchmarkData(version="lite", block_a=block_a, block_b=block_b)

        assert data1.compute_hash() == data2.compute_hash()

    def test_compute_hash_different_for_different_data(self) -> None:
        """Test that different data produces different hash."""
        block_a1 = BlockAData(
            mc_tasks=[
                MultipleChoiceTask(
                    id="mc_1",
                    category="test",
                    prompt="Q1?",
                    options=["A", "B"],
                    correct="A",
                    source="test",
                )
            ]
        )
        block_a2 = BlockAData(
            mc_tasks=[
                MultipleChoiceTask(
                    id="mc_2",
                    category="test",
                    prompt="Q2?",
                    options=["A", "B"],
                    correct="B",
                    source="test",
                )
            ]
        )

        data1 = BenchmarkData(version="lite", block_a=block_a1, block_b=BlockBData())
        data2 = BenchmarkData(version="lite", block_a=block_a2, block_b=BlockBData())

        assert data1.compute_hash() != data2.compute_hash()


class TestBenchmarkLoader:
    """Tests for BenchmarkLoader."""

    @pytest.fixture
    def temp_data_dir(self) -> Path:
        """Create temporary data directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            (data_dir / "benchmarks").mkdir()
            (data_dir / "gold").mkdir()
            yield data_dir

    def test_init_creates_directories(self, temp_data_dir: Path) -> None:
        """Test that loader initializes with data directory."""
        loader = BenchmarkLoader(temp_data_dir)
        assert loader.data_dir == temp_data_dir
        assert loader.benchmarks_dir == temp_data_dir / "benchmarks"

    def test_load_empty_benchmark(self, temp_data_dir: Path) -> None:
        """Test loading benchmark with no data files."""
        loader = BenchmarkLoader(temp_data_dir)
        benchmark = loader.load_benchmark("lite")

        # Should return empty data structures
        assert benchmark.version == "lite"
        assert benchmark.block_a.total == 0
        assert benchmark.block_b.total == 0

    def test_load_from_json(self, temp_data_dir: Path) -> None:
        """Test loading benchmark from JSON file."""
        # Create test JSON
        json_path = temp_data_dir / "test_benchmark.json"
        json_data = {
            "version": "lite",
            "block_a": {
                "mc": [
                    {
                        "id": "mc_001",
                        "type": "multiple_choice",
                        "category": "grammar",
                        "prompt": "Test question?",
                        "options": ["A) Yes", "B) No"],
                        "correct": "A",
                        "source": "test",
                    }
                ],
                "gec": [],
                "translation": [],
                "false_positive": [],
                "positive_marker": [],
            },
            "block_b": {
                "generation": [],
                "adversarial": [],
                "long_context": [],
            },
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f)

        loader = BenchmarkLoader(temp_data_dir)
        benchmark = loader.load_from_json(json_path)

        assert benchmark.version == "lite"
        assert len(benchmark.block_a.mc_tasks) == 1
        assert benchmark.block_a.mc_tasks[0].id == "mc_001"

    def test_save_and_load_benchmark(self, temp_data_dir: Path) -> None:
        """Test round-trip save and load."""
        # Create benchmark data
        block_a = BlockAData(
            mc_tasks=[
                MultipleChoiceTask(
                    id="mc_001",
                    category="test",
                    prompt="Test?",
                    options=["A", "B"],
                    correct="A",
                    source="test",
                )
            ]
        )
        benchmark = BenchmarkData(
            version="base",
            block_a=block_a,
            block_b=BlockBData(),
        )

        # Save
        output_path = temp_data_dir / "output.json"
        loader = BenchmarkLoader(temp_data_dir)
        loader.save_benchmark(benchmark, output_path)

        # Load
        loaded = loader.load_from_json(output_path)

        assert loaded.version == "base"
        assert len(loaded.block_a.mc_tasks) == 1
        assert loaded.block_a.mc_tasks[0].id == "mc_001"

    def test_get_statistics(self, temp_data_dir: Path) -> None:
        """Test statistics calculation."""
        block_a = BlockAData(
            mc_tasks=[
                MultipleChoiceTask(
                    id="mc_001",
                    category="test",
                    prompt="Test?",
                    options=["A", "B"],
                    correct="A",
                    source="test",
                )
            ],
            gec_tasks=[
                GECTask(
                    id="gec_001",
                    category="grammar",
                    input="error",
                    expected_output="correct",
                )
            ],
        )
        benchmark = BenchmarkData(
            version="lite",
            block_a=block_a,
            block_b=BlockBData(),
        )

        loader = BenchmarkLoader(temp_data_dir)
        stats = loader.get_benchmark_statistics(benchmark)

        assert stats["version"] == "lite"
        assert stats["total_tasks"] == 2
        assert stats["block_a_mc"] == 1
        assert stats["block_a_gec"] == 1

    def test_validate_benchmark_detects_duplicates(self, temp_data_dir: Path) -> None:
        """Test that validation detects duplicate IDs."""
        block_a = BlockAData(
            mc_tasks=[
                MultipleChoiceTask(
                    id="same_id",
                    category="test",
                    prompt="Q1?",
                    options=["A", "B"],
                    correct="A",
                    source="test",
                ),
                MultipleChoiceTask(
                    id="same_id",
                    category="test",
                    prompt="Q2?",
                    options=["A", "B"],
                    correct="B",
                    source="test",
                ),
            ]
        )
        benchmark = BenchmarkData(
            version="lite",
            block_a=block_a,
            block_b=BlockBData(),
        )

        loader = BenchmarkLoader(temp_data_dir)
        errors = loader.validate_benchmark(benchmark)

        assert len(errors) > 0
        assert any("Duplicate" in e for e in errors)


class TestUAGECLoader:
    """Tests for UA-GEC dataset loader."""

    def test_looks_like_russism_detection(self) -> None:
        """Test russism detection heuristic."""
        loader = UAGECLoader()

        # Should detect russism
        assert loader._looks_like_russism("прийняти участь", "взяти участь")

        # Should not detect russism
        assert not loader._looks_like_russism("правильний текст", "правильний текст")

    def test_load_from_parallel_files(self) -> None:
        """Test loading from parallel text files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            split_dir = tmppath / "train"
            split_dir.mkdir()

            # Create parallel files
            with open(split_dir / "source.txt", "w", encoding="utf-8") as f:
                f.write("Помилковий текст.\nІнший помилковий.\n")
            with open(split_dir / "target.txt", "w", encoding="utf-8") as f:
                f.write("Правильний текст.\nІнший правильний.\n")

            loader = UAGECLoader()
            docs = loader.load_from_directory(tmppath, split="train")

            assert len(docs) == 2
            assert docs[0].source_text == "Помилковий текст."
            assert docs[0].target_text == "Правильний текст."


class TestZNOLoader:
    """Tests for ZNO dataset loader."""

    def test_parse_options_from_list(self) -> None:
        """Test parsing options from list format."""
        loader = ZNOLoader()
        item = {"options": ["Варіант А", "Варіант Б", "Варіант В"]}
        options = loader._parse_options(item)
        assert len(options) == 3

    def test_parse_correct_answer_letter(self) -> None:
        """Test parsing correct answer as letter."""
        loader = ZNOLoader()
        options = ["Opt A", "Opt B", "Opt C"]

        assert loader._parse_correct_answer({"answer": "A"}, options) == "A"
        assert loader._parse_correct_answer({"answer": "b"}, options) == "B"
        assert loader._parse_correct_answer({"correct": "C"}, options) == "C"

    def test_parse_correct_answer_index(self) -> None:
        """Test parsing correct answer as index."""
        loader = ZNOLoader()
        options = ["Opt A", "Opt B", "Opt C"]

        assert loader._parse_correct_answer({"answer": 0}, options) == "A"
        assert loader._parse_correct_answer({"answer": 1}, options) == "B"
        assert loader._parse_correct_answer({"answer": 2}, options) == "C"

    def test_load_from_json(self) -> None:
        """Test loading from JSON file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(
                [
                    {
                        "question": "Яке слово написано правильно?",
                        "options": ["вірно", "правильно", "добре"],
                        "answer": 1,
                        "subject": "ukrainian_language",
                    }
                ],
                f,
                ensure_ascii=False,
            )
            json_path = Path(f.name)

        try:
            loader = ZNOLoader()
            questions = loader.load_from_json(json_path)

            assert len(questions) == 1
            assert questions[0].correct_answer == "B"
        finally:
            json_path.unlink()


class TestFLORESLoader:
    """Tests for FLORES translation loader."""

    def test_detect_traps_russian(self) -> None:
        """Test trap detection for Russian source."""
        from ukrqualbench.datasets.sources.flores import FLORESParallel

        loader = FLORESLoader()
        parallel = FLORESParallel(
            sentence_id="test",
            source_text="Мы должны принять участие в этом мероприятии.",
            source_lang="ru",
            target_text="Ми маємо взяти участь у цьому заході.",
            target_lang="uk",
        )

        traps = loader._detect_traps(parallel)
        assert len(traps) > 0
        assert "прийняти участь" in traps

    def test_create_ru_uk_trap_tasks(self) -> None:
        """Test synthetic trap task creation."""
        tasks = create_ru_uk_trap_tasks(num_tasks=5)

        assert len(tasks) == 5
        for task in tasks:
            assert task.source_lang == "ru"
            assert task.trap_type == "russism"
            assert len(task.traps) > 0

    def test_convert_to_tasks_filters_direction(self) -> None:
        """Test that convert_to_tasks filters by direction."""
        from ukrqualbench.datasets.sources.flores import FLORESParallel

        loader = FLORESLoader()
        parallels = [
            FLORESParallel(
                sentence_id="en_uk_1",
                source_text="Hello",
                source_lang="en",
                target_text="Привіт",
                target_lang="uk",
            ),
            FLORESParallel(
                sentence_id="ru_uk_1",
                source_text="Привет",
                source_lang="ru",
                target_text="Привіт",
                target_lang="uk",
            ),
        ]

        en_tasks = loader.convert_to_tasks(parallels, direction="en-uk")
        assert len(en_tasks) == 1
        assert en_tasks[0].source_lang == "en"

        ru_tasks = loader.convert_to_tasks(parallels, direction="ru-uk")
        assert len(ru_tasks) == 1
        assert ru_tasks[0].source_lang == "ru"


class TestDictionaryFiles:
    """Tests for dictionary JSON files."""

    @pytest.fixture
    def data_dir(self) -> Path:
        """Get the data directory."""
        return Path(__file__).parent.parent / "data"

    def test_russisms_dictionary_valid_json(self, data_dir: Path) -> None:
        """Test that russisms dictionary is valid JSON."""
        dict_path = data_dir / "dictionaries" / "russisms.json"
        if not dict_path.exists():
            pytest.skip("Russisms dictionary not found")

        with open(dict_path, encoding="utf-8") as f:
            data = json.load(f)

        assert "categories" in data
        assert "lexical" in data["categories"]
        assert len(data["categories"]["lexical"]["patterns"]) > 0

    def test_positive_markers_dictionary_valid_json(self, data_dir: Path) -> None:
        """Test that positive markers dictionary is valid JSON."""
        dict_path = data_dir / "dictionaries" / "positive_markers.json"
        if not dict_path.exists():
            pytest.skip("Positive markers dictionary not found")

        with open(dict_path, encoding="utf-8") as f:
            data = json.load(f)

        assert "categories" in data
        assert "vocative" in data["categories"]
        assert "particles" in data["categories"]

    def test_anglicisms_dictionary_valid_json(self, data_dir: Path) -> None:
        """Test that anglicisms dictionary is valid JSON."""
        dict_path = data_dir / "dictionaries" / "anglicisms.json"
        if not dict_path.exists():
            pytest.skip("Anglicisms dictionary not found")

        with open(dict_path, encoding="utf-8") as f:
            data = json.load(f)

        assert "categories" in data


class TestGoldStandardFiles:
    """Tests for gold standard JSON files."""

    @pytest.fixture
    def gold_dir(self) -> Path:
        """Get the gold directory."""
        return Path(__file__).parent.parent / "data" / "gold"

    def test_russisms_gold_valid(self, gold_dir: Path) -> None:
        """Test russisms gold standard file."""
        gold_path = gold_dir / "russisms.json"
        if not gold_path.exists():
            pytest.skip("Russisms gold file not found")

        with open(gold_path, encoding="utf-8") as f:
            data = json.load(f)

        assert "tasks" in data
        assert len(data["tasks"]) > 0

        # Check task structure
        task = data["tasks"][0]
        assert "id" in task
        assert "text" in task
        assert "russisms" in task

    def test_false_positives_gold_valid(self, gold_dir: Path) -> None:
        """Test false positives gold standard file."""
        gold_path = gold_dir / "false_positives.json"
        if not gold_path.exists():
            pytest.skip("False positives gold file not found")

        with open(gold_path, encoding="utf-8") as f:
            data = json.load(f)

        assert "tasks" in data
        # All should be correct
        for task in data["tasks"]:
            assert task.get("is_correct", True) is True

    def test_positive_markers_gold_valid(self, gold_dir: Path) -> None:
        """Test positive markers gold standard file."""
        gold_path = gold_dir / "positive_markers.json"
        if not gold_path.exists():
            pytest.skip("Positive markers gold file not found")

        with open(gold_path, encoding="utf-8") as f:
            data = json.load(f)

        assert "tasks" in data
        for task in data["tasks"]:
            assert "native_form" in task
            assert "non_native_forms" in task
            assert "marker_regex" in task
