"""Fixed reference corpus for the native-tokenizer fertility rate.

Fertility is measured on one Ukrainian corpus shared by every model, so the
metric reflects the tokenizer and nothing else. The corpus is the corrected
(``target``) side of the UA-GEC 2.1.3 ``gec-only`` test split, shipped inside
the ``ua-gec`` package. Documents annotated twice keep the version of the
annotator with the smallest id; documents are ordered by ``doc_id``.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

CORPUS_ID = "ua-gec-2.1.3/gec-only/test/target/min-annotator"

# Apostrophes used inside Ukrainian words: ASCII, U+2019 and U+02BC.
APOSTROPHES = "'’ʼ"

# A word is a run of word characters, optionally joined by an apostrophe or a
# hyphen ("m'yata" with any of the three apostrophes, "po-ukrainsky").
# A lone dash between spaces is not a word.
WORD_PATTERN = re.compile(rf"\w+(?:[{APOSTROPHES}-]\w+)*")


@dataclass(frozen=True)
class ReferenceCorpus:
    """Reference corpus texts plus the statistics fertility is normalised by."""

    corpus_id: str
    doc_ids: tuple[str, ...]
    texts: tuple[str, ...]
    sha256: str
    words: int
    chars: int


def count_words(text: str) -> int:
    """Count words in ``text`` with :data:`WORD_PATTERN`."""
    return len(WORD_PATTERN.findall(text))


def corpus_fingerprint(texts: Iterable[str]) -> str:
    """SHA-256 of the texts joined by the record separator (U+001E)."""
    return hashlib.sha256("\x1e".join(texts).encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def load_reference_corpus() -> ReferenceCorpus:
    """Load the UA-GEC test targets used as the fertility reference corpus."""
    from ua_gec import Corpus

    chosen: dict[str, tuple[int, str]] = {}
    for doc in Corpus(partition="test", annotation_layer="gec-only"):
        annotator = int(doc.meta.annotator_id)
        current = chosen.get(doc.doc_id)
        if current is None or annotator < current[0]:
            chosen[doc.doc_id] = (annotator, doc.target)

    doc_ids = tuple(sorted(chosen))
    texts = tuple(chosen[doc_id][1] for doc_id in doc_ids)
    return ReferenceCorpus(
        corpus_id=CORPUS_ID,
        doc_ids=doc_ids,
        texts=texts,
        sha256=corpus_fingerprint(texts),
        words=sum(count_words(text) for text in texts),
        chars=sum(len(text) for text in texts),
    )
