"""Registry of measured fertility rates, one entry per model.

Fertility depends only on the tokenizer and the fixed reference corpus, so it
is measured once and stored in ``data/fertility/registry.json``. An entry is
valid only while its own corpus fingerprint and word pattern match the current
ones; stale entries are kept on disk (a retired API model cannot be
re-measured) but :meth:`FertilityRegistry.get` does not return them.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ukrqualbench.tokenization.corpus import WORD_PATTERN, load_reference_corpus

if TYPE_CHECKING:
    from ukrqualbench.core.config import Config
    from ukrqualbench.tokenization.corpus import ReferenceCorpus

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class FertilityEntry:
    """Fertility of one model on the reference corpus."""

    model_id: str
    tokens: int
    fertility_rate: float
    chars_per_token: float
    corpus_sha256: str
    word_pattern: str
    method: str
    tokenizer_sha256: str | None
    baseline_overhead: int | None
    measured_at: str
    note: str = ""

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        del data["model_id"]
        return data

    @classmethod
    def from_json(cls, model_id: str, data: dict[str, Any]) -> FertilityEntry:
        return cls(model_id=model_id, **data)


def default_registry_path(config: Config) -> Path:
    """Registry location under the configured data directory."""
    return config.data_dir / "fertility" / "registry.json"


def fertility_block_v_fields(registry: FertilityRegistry, model_id: str) -> dict[str, Any]:
    """``block_v`` fertility fields for ``model_id``; ``None`` values when not measured."""
    entry = registry.get(model_id)
    if entry is None:
        logger.warning(
            "no fertility entry for %s; run: ukrqualbench fertility measure %s",
            model_id,
            model_id,
        )
        return {"fertility_rate": None, "fertility_method": None}
    return {"fertility_rate": entry.fertility_rate, "fertility_method": entry.method}


class FertilityRegistry:
    """Reads and writes ``registry.json``."""

    def __init__(
        self,
        path: Path,
        *,
        entries: dict[str, FertilityEntry] | None = None,
        expected_corpus_sha256: str | None = None,
    ) -> None:
        """Create a registry bound to ``path``.

        Args:
            path: JSON file location.
            entries: Entries already loaded from disk.
            expected_corpus_sha256: Fingerprint of the current corpus. ``None``
                loads the reference corpus on first validation.
        """
        self.path = path
        self._entries: dict[str, FertilityEntry] = dict(entries or {})
        self._expected_corpus_sha256 = expected_corpus_sha256

    @classmethod
    def load(cls, path: Path, *, expected_corpus_sha256: str | None = None) -> FertilityRegistry:
        """Load the registry from ``path``; a missing file gives an empty registry."""
        if not path.exists():
            return cls(path, expected_corpus_sha256=expected_corpus_sha256)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"{path}: unsupported schema_version {data.get('schema_version')!r}, "
                f"expected {SCHEMA_VERSION}"
            )
        entries = {
            model_id: FertilityEntry.from_json(model_id, raw)
            for model_id, raw in data.get("entries", {}).items()
        }
        return cls(path, entries=entries, expected_corpus_sha256=expected_corpus_sha256)

    def _current_corpus_sha256(self) -> str:
        if self._expected_corpus_sha256 is None:
            self._expected_corpus_sha256 = load_reference_corpus().sha256
        return self._expected_corpus_sha256

    def is_valid(self, entry: FertilityEntry) -> bool:
        """Whether ``entry`` was measured on the current corpus and word pattern."""
        return (
            entry.word_pattern == WORD_PATTERN.pattern
            and entry.corpus_sha256 == self._current_corpus_sha256()
        )

    def get(self, model_id: str) -> FertilityEntry | None:
        """Return the valid entry for ``model_id``, or ``None``."""
        entry = self._entries.get(model_id)
        if entry is None:
            return None
        if not self.is_valid(entry):
            logger.warning(
                "fertility entry for %s is stale (corpus or word pattern changed); "
                "re-run: ukrqualbench fertility measure %s --force",
                model_id,
                model_id,
            )
            return None
        return entry

    def all_entries(self) -> dict[str, FertilityEntry]:
        """All stored entries, including stale ones."""
        return dict(self._entries)

    def put(self, entry: FertilityEntry) -> None:
        """Add or replace the entry for ``entry.model_id``."""
        self._entries[entry.model_id] = entry

    def save(self, corpus: ReferenceCorpus) -> None:
        """Write the registry, recording ``corpus`` as the current corpus."""
        self._expected_corpus_sha256 = corpus.sha256
        data = {
            "schema_version": SCHEMA_VERSION,
            "corpus": {
                "id": corpus.corpus_id,
                "sha256": corpus.sha256,
                "docs": len(corpus.texts),
                "words": corpus.words,
                "chars": corpus.chars,
            },
            "word_pattern": WORD_PATTERN.pattern,
            "entries": {
                model_id: self._entries[model_id].to_json() for model_id in sorted(self._entries)
            },
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        tmp.replace(self.path)
