"""Native token counters.

Every counter returns the number of tokens a model's own tokenizer produces
for a plain text, without BOS/EOS or chat-template tokens.

Local backends (``hf``, ``tiktoken``) need the optional ``fertility`` extra
(``uv sync --extra fertility``); their imports are deferred so the rest of
the package works without it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from ukrqualbench.core.config import Config
    from ukrqualbench.tokenization.specs import TokenizerSpec

EXTRA_HINT = "install the optional extra: uv sync --extra fertility"


class MissingExtraError(RuntimeError):
    """A local tokenizer backend needs a package from the ``fertility`` extra."""


class ModelNotFoundError(RuntimeError):
    """The provider API does not know the model (retired or renamed ID)."""


class CalibrationError(RuntimeError):
    """The API wrapper overhead is not constant across one-token baselines."""


class TokenCounter(Protocol):
    """Counts native tokens of a plain text."""

    @property
    def method(self) -> str:
        """Short provenance label stored in the registry, e.g. ``hf:repo@sha7``."""
        ...

    @property
    def tokenizer_sha256(self) -> str | None:
        """SHA-256 of the local tokenizer file, if there is one."""
        ...

    @property
    def baseline_overhead(self) -> int | None:
        """Tokens the API adds around a message, if the backend is an API."""
        ...

    async def count(self, text: str) -> int:
        """Return the number of tokens in ``text``."""
        ...


class HFTokenCounter:
    """Counts tokens with a Hugging Face ``tokenizer.json`` pinned to a commit."""

    def __init__(
        self,
        repo: str,
        revision: str,
        *,
        hf_token: str | None = None,
        method_prefix: str = "hf",
    ) -> None:
        self._repo = repo
        self._revision = revision
        self._hf_token = hf_token
        self._method_prefix = method_prefix
        self._tokenizer: Any = None
        self._sha256: str | None = None

    @property
    def method(self) -> str:
        return f"{self._method_prefix}:{self._repo}@{self._revision[:7]}"

    @property
    def tokenizer_sha256(self) -> str | None:
        return self._sha256

    @property
    def baseline_overhead(self) -> int | None:
        return None

    def _load(self) -> Any:
        if self._tokenizer is None:
            try:
                from huggingface_hub import hf_hub_download
                from tokenizers import Tokenizer
            except ImportError as e:
                raise MissingExtraError(f"{self.method}: {EXTRA_HINT}") from e
            path = Path(
                hf_hub_download(
                    self._repo,
                    "tokenizer.json",
                    revision=self._revision,
                    token=self._hf_token,
                )
            )
            self._sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            self._tokenizer = Tokenizer.from_file(str(path))
        return self._tokenizer

    async def count(self, text: str) -> int:
        return len(self._load().encode(text, add_special_tokens=False).ids)


class TiktokenCounter:
    """Counts tokens with a named tiktoken encoding."""

    def __init__(self, encoding: str) -> None:
        self._encoding_name = encoding
        self._encoding: Any = None

    @property
    def method(self) -> str:
        return f"tiktoken:{self._encoding_name}"

    @property
    def tokenizer_sha256(self) -> str | None:
        return None

    @property
    def baseline_overhead(self) -> int | None:
        return None

    def _load(self) -> Any:
        if self._encoding is None:
            try:
                import tiktoken
            except ImportError as e:
                raise MissingExtraError(f"{self.method}: {EXTRA_HINT}") from e
            self._encoding = tiktoken.get_encoding(self._encoding_name)
        return self._encoding

    async def count(self, text: str) -> int:
        return len(self._load().encode(text, disallowed_special=()))


def make_counter(spec: TokenizerSpec, config: Config, *, is_fallback: bool = False) -> TokenCounter:
    """Build the counter described by ``spec``.

    Args:
        spec: Tokenizer spec of the model.
        config: Configuration with API keys and the Hugging Face token.
        is_fallback: Label the method ``hf-fallback`` instead of ``hf``.

    Raises:
        NotImplementedError: For API backends (added in phase 2 of the plan).
        ValueError: For an ``hf`` spec without a pinned revision.
    """
    if spec.backend == "hf":
        if spec.revision is None:
            raise ValueError(f"hf spec {spec.ref} has no pinned revision")
        return HFTokenCounter(
            spec.ref,
            spec.revision,
            hf_token=config.huggingface_token,
            method_prefix="hf-fallback" if is_fallback else "hf",
        )
    if spec.backend == "tiktoken":
        return TiktokenCounter(spec.ref)
    raise NotImplementedError(f"api backend {spec.backend}: run after phase 2")
