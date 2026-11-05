"""Native-tokenizer fertility rate: reference corpus, token counters, registry.

See ``docs/plans/native-tokenizer-fertility/plan.md`` (local) for the method.
"""

from ukrqualbench.tokenization.corpus import (
    WORD_PATTERN,
    ReferenceCorpus,
    count_words,
    load_reference_corpus,
)
from ukrqualbench.tokenization.registry import (
    FertilityEntry,
    FertilityRegistry,
    default_registry_path,
)
from ukrqualbench.tokenization.specs import MODEL_TOKENIZERS, TokenizerSpec, get_spec

__all__ = [
    "MODEL_TOKENIZERS",
    "WORD_PATTERN",
    "FertilityEntry",
    "FertilityRegistry",
    "ReferenceCorpus",
    "TokenizerSpec",
    "count_words",
    "default_registry_path",
    "get_spec",
    "load_reference_corpus",
]
