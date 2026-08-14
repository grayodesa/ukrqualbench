"""Tests for model client implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from ukrqualbench.models import (
    MODEL_PRICING,
    MockModelClient,
    ModelConfig,
    ModelResponse,
    calculate_cost,
)
from ukrqualbench.models.anthropic import AnthropicClient, create_anthropic_client
from ukrqualbench.models.google import GoogleClient, create_google_client
from ukrqualbench.models.local import (
    OllamaChatClient,
    OllamaClient,
    OpenAICompatibleClient,
    VLLMClient,
    create_ollama_client,
    create_vllm_client,
    strip_thinking_tags,
)
from ukrqualbench.models.openai import (
    AzureOpenAIClient,
    OpenAIClient,
    create_openai_client,
)

if TYPE_CHECKING:
    pass


# ============================================================================
# Base Model Tests
# ============================================================================


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = ModelConfig()

        assert config.api_key is None
        assert config.base_url is None
        assert config.timeout == 60.0
        assert config.max_retries == 3
        assert config.retry_delay == 1.0
        assert config.default_temperature == 0.0
        assert config.default_max_tokens == 8192

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = ModelConfig(
            api_key="test-key",
            base_url="https://custom.api.com",
            timeout=30.0,
            max_retries=5,
            retry_delay=2.0,
            default_temperature=0.7,
            default_max_tokens=2048,
        )

        assert config.api_key == "test-key"
        assert config.base_url == "https://custom.api.com"
        assert config.timeout == 30.0
        assert config.max_retries == 5
        assert config.retry_delay == 2.0
        assert config.default_temperature == 0.7
        assert config.default_max_tokens == 2048


class TestModelResponse:
    """Tests for ModelResponse dataclass."""

    def test_creation(self) -> None:
        """Test response creation."""
        response = ModelResponse(
            text="Test response",
            tokens_used=100,
            latency_ms=50.0,
            model_id="test-model",
            cost_usd=0.001,
        )

        assert response.text == "Test response"
        assert response.tokens_used == 100
        assert response.latency_ms == 50.0
        assert response.model_id == "test-model"
        assert response.cost_usd == 0.001
        assert response.timestamp is not None

    def test_default_cost(self) -> None:
        """Test default cost is zero."""
        response = ModelResponse(
            text="Test",
            tokens_used=10,
            latency_ms=10.0,
            model_id="test",
        )

        assert response.cost_usd == 0.0


class TestCalculateCost:
    """Tests for cost calculation."""

    def test_gpt52_cost(self) -> None:
        """Test GPT-5.2 cost calculation."""
        # GPT-5.2: $5.00/1M input, $15.00/1M output
        cost = calculate_cost("gpt-5.2", input_tokens=1000, output_tokens=500)

        expected_input = (1000 / 1_000_000) * 5.00
        expected_output = (500 / 1_000_000) * 15.00
        expected_total = expected_input + expected_output

        assert cost == pytest.approx(expected_total)

    def test_gpt4o_cost(self) -> None:
        """Test GPT-4o cost calculation (legacy)."""
        # GPT-4o: $2.50/1M input, $10.00/1M output
        cost = calculate_cost("gpt-4o", input_tokens=1000, output_tokens=500)

        expected_input = (1000 / 1_000_000) * 2.50
        expected_output = (500 / 1_000_000) * 10.00
        expected_total = expected_input + expected_output

        assert cost == pytest.approx(expected_total)

    def test_claude_opus_45_cost(self) -> None:
        """Test Claude Opus 4.5 cost calculation."""
        # Claude Opus 4.5: $5.00/1M input, $25.00/1M output
        cost = calculate_cost(
            "claude-opus-4-5-20251101",
            input_tokens=2000,
            output_tokens=1000,
        )

        expected_input = (2000 / 1_000_000) * 5.00
        expected_output = (1000 / 1_000_000) * 25.00
        expected_total = expected_input + expected_output

        assert cost == pytest.approx(expected_total)

    def test_claude_sonnet_cost(self) -> None:
        """Test Claude 3.5 Sonnet cost calculation (legacy)."""
        # Claude 3.5 Sonnet: $3.00/1M input, $15.00/1M output
        cost = calculate_cost(
            "claude-3-5-sonnet-20241022",
            input_tokens=2000,
            output_tokens=1000,
        )

        expected_input = (2000 / 1_000_000) * 3.00
        expected_output = (1000 / 1_000_000) * 15.00
        expected_total = expected_input + expected_output

        assert cost == pytest.approx(expected_total)

    def test_gemini3_cost(self) -> None:
        """Test Gemini 3 Pro cost calculation."""
        # Gemini 3 Pro: $2.00/1M input, $12.00/1M output
        cost = calculate_cost("gemini-3-pro-preview", input_tokens=5000, output_tokens=2000)

        expected_input = (5000 / 1_000_000) * 2.00
        expected_output = (2000 / 1_000_000) * 12.00
        expected_total = expected_input + expected_output

        assert cost == pytest.approx(expected_total)

    def test_gemini_cost(self) -> None:
        """Test Gemini 1.5 Pro cost calculation (legacy)."""
        # Gemini 1.5 Pro: $1.25/1M input, $5.00/1M output
        cost = calculate_cost("gemini-1.5-pro", input_tokens=5000, output_tokens=2000)

        expected_input = (5000 / 1_000_000) * 1.25
        expected_output = (2000 / 1_000_000) * 5.00
        expected_total = expected_input + expected_output

        assert cost == pytest.approx(expected_total)

    def test_local_model_free(self) -> None:
        """Test local models are free."""
        assert calculate_cost("ollama/llama3.2", 10000, 5000) == 0.0
        assert calculate_cost("vllm/mistral", 10000, 5000) == 0.0

    def test_unknown_model_free(self) -> None:
        """Test unknown models default to free."""
        assert calculate_cost("unknown-model-xyz", 10000, 5000) == 0.0


class TestModelPricing:
    """Tests for MODEL_PRICING constant."""

    def test_current_models_included(self) -> None:
        """Test that current flagship models are included in pricing."""
        # OpenAI GPT-5 family
        assert "gpt-5.2" in MODEL_PRICING
        assert "gpt-5.2-pro" in MODEL_PRICING
        assert "gpt-5-mini" in MODEL_PRICING
        # Anthropic Claude 5 family
        assert "claude-opus-5" in MODEL_PRICING
        assert "claude-sonnet-5" in MODEL_PRICING
        assert "claude-fable-5" in MODEL_PRICING
        # Anthropic Claude 4 family
        assert "claude-opus-4-8" in MODEL_PRICING
        assert "claude-opus-4-5-20251101" in MODEL_PRICING
        assert "claude-opus-4-5" in MODEL_PRICING
        assert "claude-sonnet-4" in MODEL_PRICING
        # Google Gemini 3 family
        assert "gemini-3-pro-preview" in MODEL_PRICING
        assert "gemini-3-flash-preview" in MODEL_PRICING

    def test_legacy_models_included(self) -> None:
        """Test that legacy models are still included in pricing."""
        assert "gpt-4o" in MODEL_PRICING
        assert "gpt-4o-mini" in MODEL_PRICING
        assert "claude-3-5-sonnet-20241022" in MODEL_PRICING
        assert "claude-3-5-haiku-20241022" in MODEL_PRICING
        assert "gemini-1.5-pro" in MODEL_PRICING
        assert "gemini-1.5-flash" in MODEL_PRICING

    def test_benchmarked_anthropic_models_are_priced(self) -> None:
        """Every Claude model on the published leaderboard must have a price.

        Regression guard: these four ids were absent from MODEL_PRICING, so
        `calculate_cost` silently returned 0.0 and every leaderboard run on them
        was cost-tracked at $0. A missing entry produces no error — only a wrong
        number — so the ids are pinned here explicitly.
        """
        for model_id in (
            "claude-opus-4-7",
            "claude-haiku-4-5",
            "claude-sonnet-4-5",
            "claude-3-5-haiku-latest",  # judge in the January/May runs
        ):
            assert model_id in MODEL_PRICING, f"{model_id} would be cost-tracked at $0"
            assert calculate_cost(model_id, 1_000_000, 1_000_000) > 0.0

    def test_benchmarked_gemini_models_are_priced(self) -> None:
        """Same guard for the Gemini models on the leaderboard.

        gemini-3.1-flash-lite is also the default judge, so it is billed on every
        pairwise comparison — an unpriced judge understates the cost of a whole run.
        Values are the text-tier prices from ai.google.dev, read 2026-08-15.
        """
        expected = {
            "gemini-3.7-flash": (0.75, 3.75),
            "gemini-3.5-flash": (1.50, 9.00),
            "gemini-3.1-flash-lite": (0.25, 1.50),
            "gemini-3.1-flash-lite-preview": (0.25, 1.50),
            "gemini-3-flash-preview": (0.50, 3.00),
        }
        for model_id, prices in expected.items():
            assert MODEL_PRICING.get(model_id) == prices, (
                f"{model_id}: expected {prices}, got {MODEL_PRICING.get(model_id)}"
            )

    def test_pricing_format(self) -> None:
        """Test pricing format is (input, output) tuple."""
        for model_id, pricing in MODEL_PRICING.items():
            assert isinstance(pricing, tuple), f"Pricing for {model_id} is not tuple"
            assert len(pricing) == 2, f"Pricing for {model_id} doesn't have 2 elements"
            assert all(isinstance(p, (int, float)) for p in pricing)


# ============================================================================
# Mock Model Tests
# ============================================================================


class TestMockModelClient:
    """Tests for MockModelClient."""

    @pytest.mark.asyncio
    async def test_basic_generation(self) -> None:
        """Test basic response generation."""
        client = MockModelClient()
        response = await client.generate("Test prompt")

        assert response.text == '{"result": "mock response"}'
        assert response.tokens_used == 100
        assert response.model_id == "mock-model"
        assert response.cost_usd == 0.0

    @pytest.mark.asyncio
    async def test_custom_responses(self) -> None:
        """Test custom response cycling."""
        responses = ["Response 1", "Response 2", "Response 3"]
        client = MockModelClient(responses=responses)

        r1 = await client.generate("Prompt 1")
        r2 = await client.generate("Prompt 2")
        r3 = await client.generate("Prompt 3")
        r4 = await client.generate("Prompt 4")  # Should cycle back

        assert r1.text == "Response 1"
        assert r2.text == "Response 2"
        assert r3.text == "Response 3"
        assert r4.text == "Response 1"

    @pytest.mark.asyncio
    async def test_statistics_tracking(self) -> None:
        """Test statistics tracking."""
        client = MockModelClient(tokens_per_response=50)

        await client.generate("Prompt 1")
        await client.generate("Prompt 2")
        await client.generate("Prompt 3")

        assert client.call_count == 3
        assert client.total_tokens == 150

    @pytest.mark.asyncio
    async def test_model_id_property(self) -> None:
        """Test model_id property."""
        client = MockModelClient(model_id="custom-mock")
        assert client.model_id == "custom-mock"

    @pytest.mark.asyncio
    async def test_reset_statistics(self) -> None:
        """Test statistics reset."""
        client = MockModelClient()

        await client.generate("Prompt")
        assert client.call_count == 1

        client.reset_statistics()
        assert client.call_count == 0
        assert client.total_tokens == 0
        assert client.total_cost == 0.0

    @pytest.mark.asyncio
    async def test_get_statistics(self) -> None:
        """Test get_statistics method."""
        client = MockModelClient()
        await client.generate("Prompt")

        stats = client.get_statistics()

        assert stats["model_id"] == "mock-model"
        assert stats["call_count"] == 1
        assert stats["total_tokens"] == 100
        assert stats["total_cost"] == 0.0
        assert "average_latency_ms" in stats


# ============================================================================
# OpenAI Client Tests
# ============================================================================


class TestOpenAIClient:
    """Tests for OpenAI client."""

    def test_requires_api_key(self) -> None:
        """Test that API key is required."""
        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(ValueError, match="API key required"),
        ):
            OpenAIClient()

    def test_accepts_api_key_param(self) -> None:
        """Test API key from parameter."""
        client = OpenAIClient(api_key="test-key")
        assert client._api_key == "test-key"

    def test_accepts_api_key_from_env(self) -> None:
        """Test API key from environment."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"}):
            client = OpenAIClient()
            assert client._api_key == "env-key"

    def test_model_id_property(self) -> None:
        """Test model_id property."""
        client = OpenAIClient(model_id="gpt-4o-mini", api_key="test")
        assert client.model_id == "gpt-4o-mini"

    def test_custom_base_url(self) -> None:
        """Test custom base URL."""
        client = OpenAIClient(api_key="test", base_url="https://custom.api.com")
        assert client._base_url == "https://custom.api.com"


class TestAzureOpenAIClient:
    """Tests for Azure OpenAI client."""

    def test_requires_api_key_and_endpoint(self) -> None:
        """Test that API key and endpoint are required."""
        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(ValueError, match="API key required"),
        ):
            AzureOpenAIClient(deployment_name="gpt-4o")

    def test_requires_endpoint(self) -> None:
        """Test that endpoint is required."""
        with (
            patch.dict("os.environ", {"AZURE_OPENAI_API_KEY": "test"}, clear=True),
            pytest.raises(ValueError, match="endpoint required"),
        ):
            AzureOpenAIClient(deployment_name="gpt-4o")

    def test_accepts_credentials(self) -> None:
        """Test accepting credentials."""
        client = AzureOpenAIClient(
            deployment_name="gpt-4o",
            api_key="test-key",
            azure_endpoint="https://my-resource.openai.azure.com",
        )
        assert client._api_key == "test-key"
        assert client._azure_endpoint == "https://my-resource.openai.azure.com"


class TestCreateOpenAIClient:
    """Tests for factory function."""

    def test_factory_creates_client(self) -> None:
        """Test factory function creates client."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            client = create_openai_client(
                model_id="gpt-4o",
                temperature=0.5,
                max_retries=5,
            )

            assert isinstance(client, OpenAIClient)
            assert client.model_id == "gpt-4o"
            assert client._config.default_temperature == 0.5
            assert client._config.max_retries == 5


# ============================================================================
# Anthropic Client Tests
# ============================================================================


class TestAnthropicClient:
    """Tests for Anthropic client."""

    def test_requires_api_key(self) -> None:
        """Test that API key is required."""
        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(ValueError, match="API key required"),
        ):
            AnthropicClient()

    def test_accepts_api_key_param(self) -> None:
        """Test API key from parameter."""
        client = AnthropicClient(api_key="test-key")
        assert client._api_key == "test-key"

    def test_accepts_api_key_from_env(self) -> None:
        """Test API key from environment."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "env-key"}):
            client = AnthropicClient()
            assert client._api_key == "env-key"

    def test_model_id_property(self) -> None:
        """Test model_id property."""
        client = AnthropicClient(model_id="claude-3-opus-20240229", api_key="test")
        assert client.model_id == "claude-3-opus-20240229"


class TestCreateAnthropicClient:
    """Tests for factory function."""

    def test_factory_creates_client(self) -> None:
        """Test factory function creates client."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            client = create_anthropic_client(
                model_id="claude-3-5-haiku-20241022",
                temperature=0.7,
            )

            assert isinstance(client, AnthropicClient)
            assert client.model_id == "claude-3-5-haiku-20241022"


class TestAnthropicTemperatureHandling:
    """Regression tests for the request payload Anthropic receives.

    Anthropic removed the sampling parameters starting with claude-opus-4-7;
    every model released since behaves the same way. The client sends
    `temperature` only for the models on the allowlist
    (see anthropic.py:_TEMPERATURE_SUPPORTED_PREFIXES) — anything else, known or
    unknown, gets no `temperature`, because sending it returns HTTP 400
    invalid_request_error.
    """

    @staticmethod
    def _fake_anthropic_response() -> object:
        """Build a minimal stand-in for anthropic.types.Message."""
        from unittest.mock import MagicMock

        block = MagicMock()
        block.type = "text"
        block.text = "ok"
        usage = MagicMock(input_tokens=1, output_tokens=1)
        response = MagicMock(content=[block], usage=usage)
        response.stop_reason = "end_turn"
        return response

    async def _captured_kwargs(self, model_id: str) -> dict[str, object]:
        """Call `_call_api` against a stubbed SDK and return the request kwargs."""
        from unittest.mock import AsyncMock, MagicMock

        client = AnthropicClient(model_id=model_id, api_key="test")
        mock_sdk = MagicMock()
        mock_sdk.messages.create = AsyncMock(return_value=self._fake_anthropic_response())
        client._client = mock_sdk

        await client._call_api(
            prompt="hi",
            system_prompt=None,
            temperature=0.0,
            max_tokens=100,
            json_mode=False,
        )

        return dict(mock_sdk.messages.create.call_args.kwargs)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "model_id",
        [
            "claude-3-5-haiku-20241022",
            "claude-opus-4-5",
            "claude-opus-4-5-20251101",
            "claude-opus-4-6",
            "claude-sonnet-4-20250514",
            "claude-sonnet-4-5",
            "claude-sonnet-4-6",
            "claude-haiku-4-5",
        ],
        ids=lambda m: m,
    )
    async def test_temperature_sent_for_supported_models(self, model_id: str) -> None:
        """Models that still accept sampling params must receive `temperature`."""
        kwargs = await self._captured_kwargs(model_id)

        assert kwargs["temperature"] == 0.0, (
            f"{model_id} accepts temperature — omitting it loses temperature=0 reproducibility"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "model_id",
        [
            "claude-opus-4-7",
            "claude-opus-4-8",
            "claude-opus-5",
            "claude-sonnet-5",
            "claude-fable-5",
            "claude-mythos-5",
            "claude-opus-9-future",  # unknown id: allowlist must fail safe
        ],
        ids=lambda m: m,
    )
    async def test_temperature_omitted_for_unsupported_models(self, model_id: str) -> None:
        """Models from opus-4-7 onward — and unknown ids — must NOT receive `temperature`."""
        kwargs = await self._captured_kwargs(model_id)

        assert "temperature" not in kwargs, (
            f"temperature must be omitted for {model_id} — the API rejects it with HTTP 400"
        )


class TestAnthropicRefusalHandling:
    """A safety-classifier decline must not be scored as an empty answer.

    Claude Opus 5 / Fable 5 / Mythos 5 / Sonnet 5 return HTTP 200 with
    `stop_reason: "refusal"` and empty content when their classifiers decline.
    Reading `content` blindly yields text="" — a silently wrong benchmark score.
    """

    @pytest.mark.asyncio
    async def test_refusal_raises_instead_of_returning_empty_text(self) -> None:
        from unittest.mock import AsyncMock, MagicMock

        refusal = MagicMock(content=[], usage=MagicMock(input_tokens=1, output_tokens=0))
        refusal.stop_reason = "refusal"
        refusal.stop_details = MagicMock(category="cyber")

        client = AnthropicClient(model_id="claude-fable-5", api_key="test")
        mock_sdk = MagicMock()
        mock_sdk.messages.create = AsyncMock(return_value=refusal)
        client._client = mock_sdk

        with pytest.raises(RuntimeError, match="declined the request"):
            await client._call_api(
                prompt="hi",
                system_prompt=None,
                temperature=0.0,
                max_tokens=100,
                json_mode=False,
            )


# ============================================================================
# Google Client Tests
# ============================================================================


class TestGoogleClient:
    """Tests for Google Gemini client."""

    def test_requires_api_key(self) -> None:
        """Test that API key is required."""
        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(ValueError, match="API key required"),
        ):
            GoogleClient()

    def test_accepts_api_key_param(self) -> None:
        """Test API key from parameter."""
        client = GoogleClient(api_key="test-key")
        assert client._api_key == "test-key"

    def test_accepts_api_key_from_env(self) -> None:
        """Test API key from environment (legacy GOOGLE_API_KEY)."""
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "env-key"}, clear=True):
            client = GoogleClient()
            assert client._api_key == "env-key"

    def test_accepts_gemini_api_key_from_env(self) -> None:
        """Test API key from GEMINI_API_KEY (preferred by new SDK)."""
        with patch.dict("os.environ", {"GEMINI_API_KEY": "gemini-key"}, clear=True):
            client = GoogleClient()
            assert client._api_key == "gemini-key"

    def test_gemini_api_key_takes_precedence(self) -> None:
        """Test GEMINI_API_KEY takes precedence over GOOGLE_API_KEY."""
        with patch.dict(
            "os.environ",
            {"GEMINI_API_KEY": "gemini-key", "GOOGLE_API_KEY": "google-key"},
            clear=True,
        ):
            client = GoogleClient()
            assert client._api_key == "gemini-key"

    def test_model_id_property(self) -> None:
        """Test model_id property."""
        client = GoogleClient(model_id="gemini-1.5-flash", api_key="test")
        assert client.model_id == "gemini-1.5-flash"


class TestCreateGoogleClient:
    """Tests for factory function."""

    def test_factory_creates_client(self) -> None:
        """Test factory function creates client."""
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
            client = create_google_client(
                model_id="gemini-1.5-flash",
                temperature=0.3,
            )

            assert isinstance(client, GoogleClient)
            assert client.model_id == "gemini-1.5-flash"


# ============================================================================
# Local helper Tests
# ============================================================================


class TestStripThinkingTags:
    """Tests for strip_thinking_tags helper function."""

    def test_strips_thinking_block(self) -> None:
        text = "<think>\nSome reasoning...\n</think>\n\nA"
        assert strip_thinking_tags(text) == "A"

    def test_preserves_text_without_tags(self) -> None:
        text = "Just a normal response"
        assert strip_thinking_tags(text) == "Just a normal response"

    def test_handles_multiline_after_thinking(self) -> None:
        text = "<think>Reasoning</think>\n\nLine 1\nLine 2"
        assert strip_thinking_tags(text) == "Line 1\nLine 2"

    def test_handles_empty_thinking(self) -> None:
        text = "<think></think>Answer"
        assert strip_thinking_tags(text) == "Answer"

    def test_handles_no_content_after_thinking(self) -> None:
        text = "<think>Just thinking</think>"
        assert strip_thinking_tags(text) == ""


# ============================================================================
# Local Model Tests
# ============================================================================


class TestOllamaClient:
    """Tests for Ollama client."""

    def test_default_base_url(self) -> None:
        """Test default base URL."""
        client = OllamaClient()
        assert client._base_url == "http://localhost:11434"

    def test_custom_base_url(self) -> None:
        """Test custom base URL."""
        client = OllamaClient(base_url="http://custom:11434")
        assert client._base_url == "http://custom:11434"

    def test_model_id_prefix(self) -> None:
        """Test model_id has ollama prefix."""
        client = OllamaClient(model_id="llama3.2")
        assert client.model_id == "ollama/llama3.2"

    def test_from_env(self) -> None:
        """Test base URL from environment."""
        with patch.dict("os.environ", {"OLLAMA_HOST": "http://remote:11434"}):
            client = OllamaClient()
            assert client._base_url == "http://remote:11434"


class TestOllamaChatClient:
    """Tests for Ollama chat client."""

    def test_model_id_prefix(self) -> None:
        """Test model_id has ollama prefix."""
        client = OllamaChatClient(model_id="mistral")
        assert client.model_id == "ollama/mistral"


class TestVLLMClient:
    """Tests for vLLM client."""

    def test_model_id_prefix(self) -> None:
        """Test model_id has vllm prefix."""
        client = VLLMClient(model_id="meta-llama/Llama-3.2-3B-Instruct")
        assert client.model_id == "vllm/meta-llama/Llama-3.2-3B-Instruct"

    def test_custom_base_url(self) -> None:
        """Test custom base URL."""
        client = VLLMClient(model_id="test", base_url="http://gpu-server:8000")
        assert client._base_url == "http://gpu-server:8000"


class TestOpenAICompatibleClient:
    """Tests for OpenAI-compatible client."""

    def test_model_id_prefix(self) -> None:
        """Test model_id has local prefix."""
        client = OpenAICompatibleClient(
            model_id="local-model",
            base_url="http://localhost:1234/v1",
        )
        assert client.model_id == "local/local-model"


class TestCreateOllamaClient:
    """Tests for factory function."""

    def test_creates_chat_client_by_default(self) -> None:
        """Test factory creates chat client by default."""
        client = create_ollama_client("llama3.2")
        assert isinstance(client, OllamaChatClient)

    def test_creates_base_client(self) -> None:
        """Test factory can create base client."""
        client = create_ollama_client("llama3.2", use_chat_api=False)
        assert isinstance(client, OllamaClient)


class TestCreateVLLMClient:
    """Tests for factory function."""

    def test_factory_creates_client(self) -> None:
        """Test factory function creates client."""
        client = create_vllm_client(
            model_id="mistral",
            base_url="http://localhost:8000",
            temperature=0.5,
        )

        assert isinstance(client, VLLMClient)
        assert client._vllm_model == "mistral"


# ============================================================================
# Protocol Conformance Tests
# ============================================================================


class TestProtocolConformance:
    """Tests for ModelClient protocol conformance."""

    def test_mock_client_has_model_id(self) -> None:
        """Test MockModelClient has model_id property."""
        client = MockModelClient()
        assert hasattr(client, "model_id")
        assert isinstance(client.model_id, str)

    @pytest.mark.asyncio
    async def test_mock_client_has_generate(self) -> None:
        """Test MockModelClient has async generate method."""
        client = MockModelClient()
        assert hasattr(client, "generate")

        response = await client.generate("test")
        assert isinstance(response, ModelResponse)

    def test_openai_client_has_model_id(self) -> None:
        """Test OpenAIClient has model_id property."""
        client = OpenAIClient(api_key="test")
        assert hasattr(client, "model_id")
        assert isinstance(client.model_id, str)

    def test_anthropic_client_has_model_id(self) -> None:
        """Test AnthropicClient has model_id property."""
        client = AnthropicClient(api_key="test")
        assert hasattr(client, "model_id")
        assert isinstance(client.model_id, str)

    def test_google_client_has_model_id(self) -> None:
        """Test GoogleClient has model_id property."""
        client = GoogleClient(api_key="test")
        assert hasattr(client, "model_id")
        assert isinstance(client.model_id, str)

    def test_ollama_client_has_model_id(self) -> None:
        """Test OllamaClient has model_id property."""
        client = OllamaClient()
        assert hasattr(client, "model_id")
        assert isinstance(client.model_id, str)


# ============================================================================
# OpenRouter Client Tests
# ============================================================================


class _FakeHTTPResponse:
    """Minimal stand-in for httpx.Response used in unit tests."""

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        return None


class _CapturingAsyncClient:
    """httpx.AsyncClient stub that records calls and returns canned data."""

    def __init__(
        self,
        *,
        base_url: str,
        timeout: float,
        headers: dict[str, str],
        payload: dict,
    ) -> None:
        self.base_url = base_url
        self.timeout = timeout
        self.headers = headers
        self._payload = payload
        self.last_post: tuple[str, dict] | None = None

    async def post(self, url: str, json: dict) -> _FakeHTTPResponse:
        self.last_post = (url, json)
        return _FakeHTTPResponse(self._payload)

    async def aclose(self) -> None:
        return None


class TestOpenRouterClient:
    """Tests for OpenRouterClient."""

    def test_model_id_carries_openrouter_prefix(self) -> None:
        """model_id is namespaced so it doesn't collide with local-served models."""
        from ukrqualbench.models.openrouter import OpenRouterClient

        client = OpenRouterClient(model_id="openai/gpt-5.2", api_key="sk-or-test")
        assert client.model_id == "openrouter/openai/gpt-5.2"

    def test_missing_api_key_raises(self) -> None:
        from ukrqualbench.models.openrouter import OpenRouterClient

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="OpenRouter API key required"):
                OpenRouterClient(model_id="openai/gpt-5.2")

    def test_default_headers_and_base_url(self) -> None:
        from ukrqualbench.models.openrouter import (
            OPENROUTER_DEFAULT_BASE_URL,
            OPENROUTER_DEFAULT_REFERER,
            OPENROUTER_DEFAULT_TITLE,
            OpenRouterClient,
        )

        client = OpenRouterClient(model_id="openai/gpt-5.2", api_key="sk-or-test")
        # _get_client builds the httpx.AsyncClient lazily — patch httpx so we can
        # introspect the kwargs it would receive without doing network IO.
        captured: dict = {}

        class _Spy:
            def __init__(self, **kwargs: object) -> None:
                captured.update(kwargs)

            async def aclose(self) -> None:
                return None

        with patch("httpx.AsyncClient", _Spy):
            client._get_client()

        assert captured["base_url"] == OPENROUTER_DEFAULT_BASE_URL
        headers = captured["headers"]
        assert headers["Authorization"] == "Bearer sk-or-test"
        assert headers["HTTP-Referer"] == OPENROUTER_DEFAULT_REFERER
        assert headers["X-OpenRouter-Title"] == OPENROUTER_DEFAULT_TITLE

    @pytest.mark.asyncio
    async def test_call_api_uses_usage_cost_from_response(self) -> None:
        """Cost must come from usage.cost — OpenRouter is the source of truth."""
        from ukrqualbench.models.openrouter import OpenRouterClient

        client = OpenRouterClient(model_id="openai/gpt-5.2", api_key="sk-or-test")

        stub = _CapturingAsyncClient(
            base_url="https://openrouter.ai/api/v1",
            timeout=60.0,
            headers={},
            payload={
                "id": "gen-test",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "Привіт!"},
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 4,
                    "total_tokens": 14,
                    "cost": 0.00014,
                },
                "model": "openai/gpt-5.2",
            },
        )
        client._client = stub  # bypass lazy construction

        response = await client._call_api(
            prompt="Hi",
            system_prompt="System",
            temperature=0.0,
            max_tokens=128,
            json_mode=False,
        )

        assert response.text == "Привіт!"
        assert response.tokens_used == 14
        assert response.cost_usd == pytest.approx(0.00014)

        # Verify upstream payload preserved the original (un-prefixed) model id.
        assert stub.last_post is not None
        url, body = stub.last_post
        assert url == "/chat/completions"
        assert body["model"] == "openai/gpt-5.2"
        assert body["messages"][0] == {"role": "system", "content": "System"}
        assert body["messages"][1] == {"role": "user", "content": "Hi"}

    @pytest.mark.asyncio
    async def test_missing_usage_cost_falls_back_to_zero(self) -> None:
        from ukrqualbench.models.openrouter import OpenRouterClient

        client = OpenRouterClient(model_id="openai/gpt-5.2", api_key="sk-or-test")
        stub = _CapturingAsyncClient(
            base_url="https://openrouter.ai/api/v1",
            timeout=60.0,
            headers={},
            payload={
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            },
        )
        client._client = stub

        response = await client._call_api(
            prompt="hi", system_prompt=None, temperature=0.0, max_tokens=8, json_mode=False
        )
        assert response.cost_usd == 0.0
        assert response.tokens_used == 4

    @pytest.mark.asyncio
    async def test_json_mode_sets_response_format(self) -> None:
        from ukrqualbench.models.openrouter import OpenRouterClient

        client = OpenRouterClient(model_id="openai/gpt-5.2", api_key="sk-or-test")
        stub = _CapturingAsyncClient(
            base_url="https://openrouter.ai/api/v1",
            timeout=60.0,
            headers={},
            payload={"choices": [{"message": {"content": "{}"}}], "usage": {}},
        )
        client._client = stub

        await client._call_api(
            prompt="hi", system_prompt=None, temperature=0.0, max_tokens=8, json_mode=True
        )
        assert stub.last_post is not None
        _, body = stub.last_post
        assert body["response_format"] == {"type": "json_object"}


# ============================================================================
# Retry Logic Tests
# ============================================================================


class TestRetryLogic:
    """Tests for retry logic in base client."""

    @pytest.mark.asyncio
    async def test_retry_on_failure(self) -> None:
        """Test retry on API failure."""
        client = MockModelClient()
        original_call = client._call_api
        call_count = 0

        async def failing_call(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("Simulated failure")
            return await original_call(*args, **kwargs)

        client._call_api = failing_call
        client._config.retry_delay = 0.01  # Speed up test

        response = await client.generate("test")
        assert response.text == '{"result": "mock response"}'
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self) -> None:
        """Test error when max retries exceeded."""
        client = MockModelClient()

        async def always_fail(*args, **kwargs):
            raise RuntimeError("Persistent failure")

        client._call_api = always_fail
        client._config.max_retries = 2
        client._config.retry_delay = 0.01

        with pytest.raises(RuntimeError, match="retries failed"):
            await client.generate("test")


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.asyncio
    async def test_empty_prompt(self) -> None:
        """Test handling empty prompt."""
        client = MockModelClient()
        response = await client.generate("")
        assert response.text is not None

    @pytest.mark.asyncio
    async def test_very_long_prompt(self) -> None:
        """Test handling very long prompt."""
        client = MockModelClient()
        long_prompt = "x" * 100000
        response = await client.generate(long_prompt)
        assert response.text is not None

    @pytest.mark.asyncio
    async def test_unicode_prompt(self) -> None:
        """Test handling Ukrainian text."""
        client = MockModelClient()
        response = await client.generate("Поясніть фотосинтез")
        assert response.text is not None

    @pytest.mark.asyncio
    async def test_json_mode_parameter(self) -> None:
        """Test JSON mode parameter is passed through."""
        client = MockModelClient()
        response = await client.generate("test", json_mode=True)
        assert response.text is not None

    @pytest.mark.asyncio
    async def test_system_prompt_parameter(self) -> None:
        """Test system prompt parameter."""
        client = MockModelClient()
        response = await client.generate(
            "test",
            system_prompt="You are a helpful assistant.",
        )
        assert response.text is not None

    @pytest.mark.asyncio
    async def test_temperature_zero(self) -> None:
        """Test zero temperature."""
        client = MockModelClient()
        response = await client.generate("test", temperature=0.0)
        assert response.text is not None

    @pytest.mark.asyncio
    async def test_temperature_max(self) -> None:
        """Test max temperature."""
        client = MockModelClient()
        response = await client.generate("test", temperature=2.0)
        assert response.text is not None


# ============================================================================
# Integration with Judges
# ============================================================================


class TestJudgeIntegration:
    """Tests for integration with judge system."""

    @pytest.mark.asyncio
    async def test_mock_client_works_with_pairwise_judge(self) -> None:
        """Test MockModelClient works with PairwiseJudge."""
        from ukrqualbench.judges import PairwiseJudge

        mock_response = """
        {
            "winner": "A",
            "confidence": "high",
            "reasoning": "Response A is better."
        }
        """

        client = MockModelClient(responses=[mock_response])
        judge = PairwiseJudge(model=client, shuffle_positions=False)

        verdict = await judge.judge_pairwise(
            prompt="Поясніть фотосинтез",
            response_a="Фотосинтез - це процес...",
            response_b="Фотосинтез є процесом...",
        )

        assert verdict.winner.value == "A"
        assert verdict.confidence.value == "high"
