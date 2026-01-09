"""Benchmark data loader and assembler.

Loads and assembles benchmark datasets from multiple sources,
creating lite, base, and large benchmark versions.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

from pydantic import BaseModel

from ukrqualbench.core.schemas import (
    AdversarialTask,
    FalsePositiveTask,
    FreeGenerationTask,
    GECTask,
    LongContextTask,
    MultipleChoiceTask,
    PositiveMarkerTask,
    TranslationTask,
)

if TYPE_CHECKING:
    pass

# Type aliases for task types
BlockATask: TypeAlias = (
    MultipleChoiceTask | GECTask | TranslationTask | FalsePositiveTask | PositiveMarkerTask
)
BlockBTask: TypeAlias = FreeGenerationTask | AdversarialTask | LongContextTask

BenchmarkVersionType: TypeAlias = Literal["lite", "base", "large"]


# Benchmark version specifications
BENCHMARK_SPECS: dict[str, dict[str, dict[str, int]]] = {
    "lite": {
        "block_a": {
            "mc": 100,
            "gec": 50,
            "translation": 30,
            "false_positive": 10,
            "positive_marker": 10,
        },
        "block_b": {
            "free_generation": 70,
            "adversarial": 20,
            "long_context": 10,
        },
    },
    "base": {
        "block_a": {
            "mc": 200,
            "gec": 200,
            "translation": 100,
            "false_positive": 25,
            "positive_marker": 25,
        },
        "block_b": {
            "free_generation": 150,
            "adversarial": 60,
            "long_context": 40,
        },
    },
    "large": {
        "block_a": {
            "mc": 400,
            "gec": 400,
            "translation": 200,
            "false_positive": 50,
            "positive_marker": 50,
        },
        "block_b": {
            "free_generation": 300,
            "adversarial": 100,
            "long_context": 50,
        },
    },
}


class BenchmarkMetadata(BaseModel):
    """Metadata for a benchmark dataset."""

    version: BenchmarkVersionType
    dataset_hash: str
    total_block_a: int
    total_block_b: int
    created_at: str
    sources: list[str]


@dataclass
class BlockAData:
    """Container for Block A (calibration) tasks."""

    mc_tasks: list[MultipleChoiceTask] = field(default_factory=list)
    gec_tasks: list[GECTask] = field(default_factory=list)
    translation_tasks: list[TranslationTask] = field(default_factory=list)
    false_positive_tasks: list[FalsePositiveTask] = field(default_factory=list)
    positive_marker_tasks: list[PositiveMarkerTask] = field(default_factory=list)

    @property
    def total(self) -> int:
        """Total number of Block A tasks."""
        return (
            len(self.mc_tasks)
            + len(self.gec_tasks)
            + len(self.translation_tasks)
            + len(self.false_positive_tasks)
            + len(self.positive_marker_tasks)
        )

    def all_tasks(self) -> list[BlockATask]:
        """Get all Block A tasks as a flat list."""
        tasks: list[BlockATask] = []
        tasks.extend(self.mc_tasks)
        tasks.extend(self.gec_tasks)
        tasks.extend(self.translation_tasks)
        tasks.extend(self.false_positive_tasks)
        tasks.extend(self.positive_marker_tasks)
        return tasks


@dataclass
class BlockBData:
    """Container for Block B (generation) tasks."""

    generation_tasks: list[FreeGenerationTask] = field(default_factory=list)
    adversarial_tasks: list[AdversarialTask] = field(default_factory=list)
    long_context_tasks: list[LongContextTask] = field(default_factory=list)

    @property
    def total(self) -> int:
        """Total number of Block B tasks."""
        return (
            len(self.generation_tasks) + len(self.adversarial_tasks) + len(self.long_context_tasks)
        )

    def all_tasks(self) -> list[BlockBTask]:
        """Get all Block B tasks as a flat list."""
        tasks: list[BlockBTask] = []
        tasks.extend(self.generation_tasks)
        tasks.extend(self.adversarial_tasks)
        tasks.extend(self.long_context_tasks)
        return tasks


@dataclass
class BenchmarkData:
    """Complete benchmark dataset."""

    version: BenchmarkVersionType
    block_a: BlockAData
    block_b: BlockBData
    metadata: BenchmarkMetadata | None = None

    @property
    def total_tasks(self) -> int:
        """Total number of all tasks."""
        return self.block_a.total + self.block_b.total

    def compute_hash(self) -> str:
        """Compute SHA-256 hash of benchmark data for reproducibility."""
        # Serialize all tasks to JSON
        data = {
            "version": self.version,
            "block_a": [t.model_dump_json() for t in self.block_a.all_tasks()],
            "block_b": [t.model_dump_json() for t in self.block_b.all_tasks()],
        }
        json_str = json.dumps(data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(json_str.encode()).hexdigest()


class BenchmarkLoader:
    """Loader for benchmark datasets.

    Loads tasks from JSON files and assembles complete benchmark
    datasets according to version specifications.
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        """Initialize loader.

        Args:
            data_dir: Path to data directory containing benchmarks.
        """
        self.data_dir = data_dir or Path("data")
        self.benchmarks_dir = self.data_dir / "benchmarks"
        self.gold_dir = self.data_dir / "gold"

    def load_benchmark(self, version: BenchmarkVersionType = "base") -> BenchmarkData:
        """Load a complete benchmark dataset.

        Args:
            version: Benchmark version (lite, base, large).

        Returns:
            Complete benchmark data.
        """
        spec = BENCHMARK_SPECS[version]

        # Load Block A
        block_a = BlockAData(
            mc_tasks=self._load_mc_tasks(spec["block_a"]["mc"]),
            gec_tasks=self._load_gec_tasks(spec["block_a"]["gec"]),
            translation_tasks=self._load_translation_tasks(spec["block_a"]["translation"]),
            false_positive_tasks=self._load_false_positive_tasks(spec["block_a"]["false_positive"]),
            positive_marker_tasks=self._load_positive_marker_tasks(
                spec["block_a"]["positive_marker"]
            ),
        )

        # Load Block B
        block_b = BlockBData(
            generation_tasks=self._load_generation_tasks(spec["block_b"]["free_generation"]),
            adversarial_tasks=self._load_adversarial_tasks(spec["block_b"]["adversarial"]),
            long_context_tasks=self._load_long_context_tasks(spec["block_b"]["long_context"]),
        )

        benchmark = BenchmarkData(version=version, block_a=block_a, block_b=block_b)

        # Compute metadata
        from datetime import datetime

        benchmark.metadata = BenchmarkMetadata(
            version=version,
            dataset_hash=benchmark.compute_hash(),
            total_block_a=block_a.total,
            total_block_b=block_b.total,
            created_at=datetime.now().isoformat(),
            sources=["UA-GEC", "ZNO", "FLORES", "Brown-UK"],
        )

        return benchmark

    def load_from_json(self, json_path: Path) -> BenchmarkData:
        """Load benchmark from pre-assembled JSON file.

        Args:
            json_path: Path to benchmark JSON file.

        Returns:
            Benchmark data.
        """
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        version = data.get("version", "base")
        block_a_data = data.get("block_a", {})
        block_b_data = data.get("block_b", {})

        block_a = BlockAData(
            mc_tasks=[MultipleChoiceTask.model_validate(t) for t in block_a_data.get("mc", [])],
            gec_tasks=[GECTask.model_validate(t) for t in block_a_data.get("gec", [])],
            translation_tasks=[
                TranslationTask.model_validate(t) for t in block_a_data.get("translation", [])
            ],
            false_positive_tasks=[
                FalsePositiveTask.model_validate(t) for t in block_a_data.get("false_positive", [])
            ],
            positive_marker_tasks=[
                PositiveMarkerTask.model_validate(t)
                for t in block_a_data.get("positive_marker", [])
            ],
        )

        block_b = BlockBData(
            generation_tasks=[
                FreeGenerationTask.model_validate(t) for t in block_b_data.get("generation", [])
            ],
            adversarial_tasks=[
                AdversarialTask.model_validate(t) for t in block_b_data.get("adversarial", [])
            ],
            long_context_tasks=[
                LongContextTask.model_validate(t) for t in block_b_data.get("long_context", [])
            ],
        )

        return BenchmarkData(version=version, block_a=block_a, block_b=block_b)

    def save_benchmark(self, benchmark: BenchmarkData, output_path: Path) -> None:
        """Save benchmark to JSON file.

        Args:
            benchmark: Benchmark data to save.
            output_path: Output file path.
        """
        data = {
            "version": benchmark.version,
            "metadata": benchmark.metadata.model_dump() if benchmark.metadata else None,
            "block_a": {
                "mc": [t.model_dump() for t in benchmark.block_a.mc_tasks],
                "gec": [t.model_dump() for t in benchmark.block_a.gec_tasks],
                "translation": [t.model_dump() for t in benchmark.block_a.translation_tasks],
                "false_positive": [t.model_dump() for t in benchmark.block_a.false_positive_tasks],
                "positive_marker": [
                    t.model_dump() for t in benchmark.block_a.positive_marker_tasks
                ],
            },
            "block_b": {
                "generation": [t.model_dump() for t in benchmark.block_b.generation_tasks],
                "adversarial": [t.model_dump() for t in benchmark.block_b.adversarial_tasks],
                "long_context": [t.model_dump() for t in benchmark.block_b.long_context_tasks],
            },
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # =========================================================================
    # Private loading methods
    # =========================================================================

    def _load_json_tasks(self, filename: str) -> list[dict[str, Any]]:
        """Load tasks from JSON file in benchmarks directory."""
        path = self.benchmarks_dir / filename
        if not path.exists():
            return []
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else data.get("tasks", [])

    def _load_mc_tasks(self, count: int) -> list[MultipleChoiceTask]:
        """Load multiple choice tasks."""
        raw_tasks = self._load_json_tasks("mc_tasks.json")
        tasks = [MultipleChoiceTask.model_validate(t) for t in raw_tasks[:count]]
        return tasks

    def _load_gec_tasks(self, count: int) -> list[GECTask]:
        """Load GEC tasks."""
        raw_tasks = self._load_json_tasks("gec_tasks.json")
        tasks = [GECTask.model_validate(t) for t in raw_tasks[:count]]
        return tasks

    def _load_translation_tasks(self, count: int) -> list[TranslationTask]:
        """Load translation tasks."""
        raw_tasks = self._load_json_tasks("translation_tasks.json")
        tasks = [TranslationTask.model_validate(t) for t in raw_tasks[:count]]
        return tasks

    def _load_false_positive_tasks(self, count: int) -> list[FalsePositiveTask]:
        """Load false positive tasks."""
        raw_tasks = self._load_json_tasks("false_positive_tasks.json")
        tasks = [FalsePositiveTask.model_validate(t) for t in raw_tasks[:count]]
        return tasks

    def _load_positive_marker_tasks(self, count: int) -> list[PositiveMarkerTask]:
        """Load positive marker tasks."""
        raw_tasks = self._load_json_tasks("positive_marker_tasks.json")
        tasks = [PositiveMarkerTask.model_validate(t) for t in raw_tasks[:count]]
        return tasks

    def _load_generation_tasks(self, count: int) -> list[FreeGenerationTask]:
        """Load free generation tasks."""
        raw_tasks = self._load_json_tasks("generation_tasks.json")
        tasks = [FreeGenerationTask.model_validate(t) for t in raw_tasks[:count]]
        return tasks

    def _load_adversarial_tasks(self, count: int) -> list[AdversarialTask]:
        """Load adversarial tasks."""
        raw_tasks = self._load_json_tasks("adversarial_tasks.json")
        tasks = [AdversarialTask.model_validate(t) for t in raw_tasks[:count]]
        return tasks

    def _load_long_context_tasks(self, count: int) -> list[LongContextTask]:
        """Load long context tasks."""
        raw_tasks = self._load_json_tasks("long_context_tasks.json")
        tasks = [LongContextTask.model_validate(t) for t in raw_tasks[:count]]
        return tasks

    # =========================================================================
    # Statistics and validation
    # =========================================================================

    def get_benchmark_statistics(self, benchmark: BenchmarkData) -> dict[str, int | float | str]:
        """Get statistics about a benchmark dataset.

        Args:
            benchmark: Benchmark data.

        Returns:
            Dictionary with statistics.
        """
        return {
            "version": benchmark.version,
            "total_tasks": benchmark.total_tasks,
            "block_a_total": benchmark.block_a.total,
            "block_a_mc": len(benchmark.block_a.mc_tasks),
            "block_a_gec": len(benchmark.block_a.gec_tasks),
            "block_a_translation": len(benchmark.block_a.translation_tasks),
            "block_a_false_positive": len(benchmark.block_a.false_positive_tasks),
            "block_a_positive_marker": len(benchmark.block_a.positive_marker_tasks),
            "block_b_total": benchmark.block_b.total,
            "block_b_generation": len(benchmark.block_b.generation_tasks),
            "block_b_adversarial": len(benchmark.block_b.adversarial_tasks),
            "block_b_long_context": len(benchmark.block_b.long_context_tasks),
            "dataset_hash": benchmark.metadata.dataset_hash if benchmark.metadata else "",
        }

    def validate_benchmark(self, benchmark: BenchmarkData) -> list[str]:
        """Validate benchmark data against specifications.

        Args:
            benchmark: Benchmark data to validate.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors: list[str] = []
        spec = BENCHMARK_SPECS[benchmark.version]

        # Check Block A counts
        block_a_spec = spec["block_a"]
        if len(benchmark.block_a.mc_tasks) < block_a_spec["mc"]:
            errors.append(f"MC tasks: {len(benchmark.block_a.mc_tasks)} < {block_a_spec['mc']}")
        if len(benchmark.block_a.gec_tasks) < block_a_spec["gec"]:
            errors.append(f"GEC tasks: {len(benchmark.block_a.gec_tasks)} < {block_a_spec['gec']}")

        # Check Block B counts
        block_b_spec = spec["block_b"]
        if len(benchmark.block_b.generation_tasks) < block_b_spec["free_generation"]:
            errors.append(
                f"Generation tasks: {len(benchmark.block_b.generation_tasks)} < "
                f"{block_b_spec['free_generation']}"
            )

        # Check for duplicate IDs
        all_ids: set[str] = set()
        for task_a in benchmark.block_a.all_tasks():
            if task_a.id in all_ids:
                errors.append(f"Duplicate task ID: {task_a.id}")
            all_ids.add(task_a.id)
        for task_b in benchmark.block_b.all_tasks():
            if task_b.id in all_ids:
                errors.append(f"Duplicate task ID: {task_b.id}")
            all_ids.add(task_b.id)

        return errors
