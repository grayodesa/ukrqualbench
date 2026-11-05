"""Native token counters.

Every counter returns the number of tokens a model's own tokenizer produces
for a plain text, without BOS/EOS or chat-template tokens.

Local backends (``hf``, ``tiktoken``) need the optional ``fertility`` extra
(``uv sync --extra fertility``); their imports are deferred so the rest of
the package works without it.
"""

from __future__ import annotations

import asyncio
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

    async def aclose(self) -> None:
        """Release network clients; call inside the event loop that used them."""
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

    async def aclose(self) -> None:
        return None


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

    async def aclose(self) -> None:
        return None


class APITokenCounter:
    """Counts tokens through a provider endpoint, minus the message wrapper.

    Provider count endpoints measure a whole request: role markers and message
    boundaries are included. The wrapper overhead ``O`` is measured once from
    one-token texts (``count(".") - 1``) and subtracted from every count. The
    same overhead must come out of ``"."``, ``"a"`` and ``"x"``; otherwise the
    scheme does not hold for this model and :class:`CalibrationError` is raised.
    Baseline and document requests carry identical parameters (no system,
    thinking, reasoning or tools), so the wrapper is the same for both.

    Lone characters are not equally cheap in every tokenizer (count_tokens,
    2026-11-05): pre-4.7 Claude spends two tokens on a digit, Claude Opus 4.7
    on ``,`` ``;`` digits and capitals, Claude 5 on a capital. ``.``, ``a`` and
    ``x`` are one token in all of them, in o200k and in Gemma.
    """

    provider = "api"
    BASELINES: tuple[str, ...] = (".", "a", "x")

    def __init__(self, model: str) -> None:
        self.model = model
        self._overhead: int | None = None
        self._overhead_lock = asyncio.Lock()

    @property
    def method(self) -> str:
        return f"api:{self.provider}"

    @property
    def tokenizer_sha256(self) -> str | None:
        return None

    @property
    def baseline_overhead(self) -> int | None:
        return self._overhead

    async def raw_count(self, text: str) -> int:
        """Token count of a one-message request containing ``text``."""
        raise NotImplementedError

    async def overhead(self) -> int:
        """Wrapper overhead of this model, measured on first use."""
        async with self._overhead_lock:
            if self._overhead is None:
                raws = {b: await self.raw_count(b) for b in self.BASELINES}
                overheads = {raw - 1 for raw in raws.values()}
                if len(overheads) != 1:
                    raise CalibrationError(
                        f"{self.method} {self.model}: one-token baselines give different "
                        f"request sizes {raws}; wrapper overhead is not constant"
                    )
                self._overhead = overheads.pop()
            return self._overhead

    async def count(self, text: str) -> int:
        overhead = await self.overhead()
        return await self.raw_count(text) - overhead

    async def aclose(self) -> None:
        return None


class AnthropicTokenCounter(APITokenCounter):
    """``POST /v1/messages/count_tokens`` (free; the docs call it an estimate)."""

    provider = "anthropic"

    def __init__(self, model: str, api_key: str, *, max_retries: int, timeout: float) -> None:
        super().__init__(model)
        import anthropic

        self._anthropic = anthropic
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key, max_retries=max_retries, timeout=timeout
        )

    async def raw_count(self, text: str) -> int:
        try:
            result = await self._client.messages.count_tokens(
                model=self.model, messages=[{"role": "user", "content": text}]
            )
        except self._anthropic.NotFoundError as e:
            raise ModelNotFoundError(f"{self.model}: {e}") from e
        return int(result.input_tokens)

    async def aclose(self) -> None:
        await self._client.close()


class OpenAITokenCounter(APITokenCounter):
    """``POST /v1/responses/input_tokens``."""

    provider = "openai"

    def __init__(self, model: str, api_key: str, *, max_retries: int, timeout: float) -> None:
        super().__init__(model)
        import openai

        self._openai = openai
        self._client = openai.AsyncOpenAI(api_key=api_key, max_retries=max_retries, timeout=timeout)

    async def raw_count(self, text: str) -> int:
        try:
            result = await self._client.responses.input_tokens.count(model=self.model, input=text)
        except self._openai.NotFoundError as e:
            raise ModelNotFoundError(f"{self.model}: {e}") from e
        except self._openai.BadRequestError as e:
            if getattr(e, "code", None) == "model_not_found":
                raise ModelNotFoundError(f"{self.model}: {e}") from e
            raise
        return int(result.input_tokens)

    async def aclose(self) -> None:
        await self._client.close()


class GeminiTokenCounter(APITokenCounter):
    """``models.countTokens`` of the Gemini API (free)."""

    provider = "gemini"

    def __init__(self, model: str, api_key: str, *, max_retries: int) -> None:
        super().__init__(model)
        from google import genai
        from google.genai import errors, types

        self._errors = errors
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(attempts=max(1, max_retries))
            ),
        )

    async def raw_count(self, text: str) -> int:
        try:
            result = await self._client.aio.models.count_tokens(model=self.model, contents=text)
        except self._errors.ClientError as e:
            if e.code == 404:
                raise ModelNotFoundError(f"{self.model}: {e}") from e
            raise
        if result.total_tokens is None:
            raise RuntimeError(f"{self.model}: countTokens returned no total_tokens")
        return int(result.total_tokens)

    async def aclose(self) -> None:
        await self._client.aio.aclose()


def _require_key(key: str | None, backend: str, env_name: str) -> str:
    if not key:
        raise ValueError(f"{backend} API key required: set UKRQUALBENCH_{env_name} or {env_name}")
    return key


def make_counter(spec: TokenizerSpec, config: Config, *, is_fallback: bool = False) -> TokenCounter:
    """Build the counter described by ``spec``.

    Args:
        spec: Tokenizer spec of the model.
        config: Configuration with API keys and the Hugging Face token.
        is_fallback: Label the method ``hf-fallback`` instead of ``hf``.

    Raises:
        ValueError: An ``hf`` spec without a pinned revision, or a missing API key.
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
    if spec.backend == "anthropic":
        return AnthropicTokenCounter(
            spec.ref,
            _require_key(config.anthropic_api_key, "Anthropic", "ANTHROPIC_API_KEY"),
            max_retries=config.max_retries,
            timeout=float(config.request_timeout),
        )
    if spec.backend == "openai":
        return OpenAITokenCounter(
            spec.ref,
            _require_key(config.openai_api_key, "OpenAI", "OPENAI_API_KEY"),
            max_retries=config.max_retries,
            timeout=float(config.request_timeout),
        )
    if spec.backend == "gemini":
        return GeminiTokenCounter(
            spec.ref,
            _require_key(config.google_api_key, "Google", "GOOGLE_API_KEY"),
            max_retries=config.max_retries,
        )
    raise ValueError(f"unknown tokenizer backend {spec.backend!r}")
