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
    NebiusClient,
    OllamaChatClient,
    OllamaClient,
    OpenAICompatibleClient,
    VLLMClient,
    create_nebius_client,
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
        # Anthropic Claude 4 family
        assert "claude-opus-4-5-20251101" in MODEL_PRICING
        assert "claude-opus-4-5" in MODEL_PRICING
        assert "claude-sonnet-4" in MODEL_PRICING
        # Google Gemini 3 family
        assert "gemini-3-pro-preview" in MODEL_PRICING
        assert "gemini-3-flash-preview" in MODEL_PRICING
        # Nebius Token Factory models
        assert "nebius/deepseek-ai/DeepSeek-R1-0528" in MODEL_PRICING
        assert "nebius/deepseek-ai/DeepSeek-V3" in MODEL_PRICING
        assert "nebius/Qwen/Qwen3-235B-Instruct" in MODEL_PRICING
        assert "nebius/meta-llama/Llama-3.3-70B-Instruct" in MODEL_PRICING

    def test_legacy_models_included(self) -> None:
        """Test that legacy models are still included in pricing."""
        assert "gpt-4o" in MODEL_PRICING
        assert "gpt-4o-mini" in MODEL_PRICING
        assert "claude-3-5-sonnet-20241022" in MODEL_PRICING
        assert "claude-3-5-haiku-20241022" in MODEL_PRICING
        assert "gemini-1.5-pro" in MODEL_PRICING
        assert "gemini-1.5-flash" in MODEL_PRICING

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
# Nebius Token Factory Tests
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


class TestNebiusClient:
    """Tests for Nebius Token Factory client."""

    def test_requires_api_key(self) -> None:
        """Test that API key is required."""
        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(ValueError, match="API key required"),
        ):
            NebiusClient()

    def test_accepts_api_key_param(self) -> None:
        """Test API key from parameter."""
        client = NebiusClient(api_key="test-key")
        assert client._api_key == "test-key"

    def test_accepts_api_key_from_env(self) -> None:
        """Test API key from environment."""
        with patch.dict("os.environ", {"NEBIUS_API_KEY": "env-key"}):
            client = NebiusClient()
            assert client._api_key == "env-key"

    def test_model_id_property(self) -> None:
        """Test model_id property has nebius prefix."""
        client = NebiusClient(model_id="deepseek-ai/DeepSeek-V3", api_key="test")
        assert client.model_id == "nebius/deepseek-ai/DeepSeek-V3"

    def test_default_base_url(self) -> None:
        """Test default base URL."""
        client = NebiusClient(api_key="test")
        assert client._base_url == "https://api.tokenfactory.nebius.com/v1"

    def test_custom_base_url(self) -> None:
        """Test custom base URL."""
        client = NebiusClient(api_key="test", base_url="https://custom.api.com/v1")
        assert client._base_url == "https://custom.api.com/v1"


class TestCreateNebiusClient:
    """Tests for factory function."""

    def test_factory_creates_client(self) -> None:
        """Test factory function creates client."""
        with patch.dict("os.environ", {"NEBIUS_API_KEY": "test-key"}):
            client = create_nebius_client(
                model_id="deepseek-ai/DeepSeek-R1-0528",
                temperature=0.5,
            )

            assert isinstance(client, NebiusClient)
            assert client.model_id == "nebius/deepseek-ai/DeepSeek-R1-0528"
            assert client._config.default_temperature == 0.5

    def test_factory_default_model(self) -> None:
        """Test factory uses default model."""
        with patch.dict("os.environ", {"NEBIUS_API_KEY": "test-key"}):
            client = create_nebius_client()

            assert "DeepSeek-R1-0528" in client.model_id


class TestNebiusCostCalculation:
    """Tests for Nebius model cost calculation."""

    def test_deepseek_r1_cost(self) -> None:
        """Test DeepSeek-R1 cost calculation."""
        # DeepSeek-R1-0528: $0.80/1M input, $2.40/1M output
        cost = calculate_cost(
            "nebius/deepseek-ai/DeepSeek-R1-0528",
            input_tokens=10000,
            output_tokens=5000,
        )

        expected_input = (10000 / 1_000_000) * 0.80
        expected_output = (5000 / 1_000_000) * 2.40
        expected_total = expected_input + expected_output

        assert cost == pytest.approx(expected_total)

    def test_qwen_cost(self) -> None:
        """Test Qwen model cost calculation."""
        # Qwen3-235B-Instruct: $0.20/1M input, $0.60/1M output
        cost = calculate_cost(
            "nebius/Qwen/Qwen3-235B-Instruct",
            input_tokens=5000,
            output_tokens=2000,
        )

        expected_input = (5000 / 1_000_000) * 0.20
        expected_output = (2000 / 1_000_000) * 0.60
        expected_total = expected_input + expected_output

        assert cost == pytest.approx(expected_total)

    def test_llama_cost(self) -> None:
        """Test Llama model cost calculation."""
        # Llama-3.3-70B: $0.13/1M input, $0.40/1M output
        cost = calculate_cost(
            "nebius/meta-llama/Llama-3.3-70B-Instruct",
            input_tokens=8000,
            output_tokens=4000,
        )

        expected_input = (8000 / 1_000_000) * 0.13
        expected_output = (4000 / 1_000_000) * 0.40
        expected_total = expected_input + expected_output

        assert cost == pytest.approx(expected_total)


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

    def test_nebius_client_has_model_id(self) -> None:
        """Test NebiusClient has model_id property."""
        client = NebiusClient(api_key="test")
        assert hasattr(client, "model_id")
        assert isinstance(client.model_id, str)
        assert client.model_id.startswith("nebius/")


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
