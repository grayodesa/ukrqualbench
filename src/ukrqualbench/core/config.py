"""Configuration system for UkrQualBench using Pydantic Settings.

Configuration can be set via:
1. Environment variables (prefixed with UKRQUALBENCH_)
2. .env file
3. Direct instantiation
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class BenchmarkVersion(str, Enum):
    """Available benchmark versions with different task counts."""

    LITE = "lite"
    BASE = "base"
    LARGE = "large"


BENCHMARK_SIZES: dict[BenchmarkVersion, dict[str, int]] = {
    BenchmarkVersion.LITE: {"block_a": 200, "block_b": 100},
    BenchmarkVersion.BASE: {"block_a": 550, "block_b": 250},
    BenchmarkVersion.LARGE: {"block_a": 1100, "block_b": 450},
}


class Config(BaseSettings):
    """Main configuration for UkrQualBench.

    All settings can be overridden via environment variables with
    the UKRQUALBENCH_ prefix (e.g., UKRQUALBENCH_OPENAI_API_KEY).
    """

    model_config = SettingsConfigDict(
        env_prefix="UKRQUALBENCH_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Keys - Primary providers
    # Use AliasChoices to accept both prefixed (UKRQUALBENCH_*) and standard env vars
    openai_api_key: str | None = Field(  # type: ignore[pydantic-alias]
        default=None,
        description="OpenAI API key for GPT models",
        validation_alias=AliasChoices(
            "UKRQUALBENCH_OPENAI_API_KEY",
            "OPENAI_API_KEY",
        ),
    )
    anthropic_api_key: str | None = Field(  # type: ignore[pydantic-alias]
        default=None,
        description="Anthropic API key for Claude models",
        validation_alias=AliasChoices(
            "UKRQUALBENCH_ANTHROPIC_API_KEY",
            "ANTHROPIC_API_KEY",
        ),
    )
    google_api_key: str | None = Field(  # type: ignore[pydantic-alias]
        default=None,
        description="Google AI API key for Gemini models",
        validation_alias=AliasChoices(
            "UKRQUALBENCH_GOOGLE_API_KEY",
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
        ),
    )

    # Azure OpenAI (optional)
    azure_openai_api_key: str | None = Field(
        default=None,
        description="Azure OpenAI API key",
    )
    azure_openai_endpoint: str | None = Field(
        default=None,
        description="Azure OpenAI endpoint URL",
    )

    # Google Cloud / Vertex AI (optional)
    google_cloud_project: str | None = Field(
        default=None,
        description="Google Cloud project ID for Vertex AI",
    )

    # Nebius Token Factory (optional)
    nebius_api_key: str | None = Field(  # type: ignore[pydantic-alias]
        default=None,
        description="Nebius API key for Token Factory models",
        validation_alias=AliasChoices(
            "UKRQUALBENCH_NEBIUS_API_KEY",
            "NEBIUS_API_KEY",
        ),
    )

    # HuggingFace (for gated datasets)
    huggingface_token: str | None = Field(  # type: ignore[pydantic-alias]
        default=None,
        description="HuggingFace token for accessing gated datasets",
        validation_alias=AliasChoices(
            "UKRQUALBENCH_HUGGINGFACE_TOKEN",
            "HUGGINGFACE_TOKEN",
            "HF_TOKEN",
        ),
    )

    # Local models (LM Studio, Ollama, vLLM, etc. - OpenAI-compatible)
    local_base_url: str | None = Field(  # type: ignore[pydantic-alias]
        default=None,
        description="Local model API base URL (OpenAI-compatible, e.g., http://localhost:1234/v1)",
        validation_alias=AliasChoices(
            "UKRQUALBENCH_LOCAL_BASE_URL",
            "UKRQUALBENCH_OLLAMA_BASE_URL",
        ),
    )
    vllm_api_key: str | None = Field(
        default=None,
        description="vLLM API key (if authentication required)",
    )
    vllm_base_url: str = Field(
        default="http://localhost:8000",
        description="vLLM API base URL",
    )

    # Benchmark settings
    benchmark_version: BenchmarkVersion = Field(
        default=BenchmarkVersion.BASE,
        description="Benchmark version: lite (~30min), base (~2hr), large (~5hr)",
    )
    default_judge: str = Field(
        default="claude-3-5-haiku-latest",
        description="Default judge model for pairwise comparisons",
    )

    # ELO settings
    elo_initial_rating: int = Field(
        default=1500,
        ge=1000,
        le=2000,
        description="Initial ELO rating for all models",
    )
    elo_k_factor: int = Field(
        default=32,
        ge=1,
        le=100,
        description="K-factor for ELO rating updates",
    )

    # Execution settings
    max_concurrent_requests: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum concurrent API requests per provider (NFR-6)",
    )
    request_timeout: int = Field(
        default=60,
        ge=10,
        le=300,
        description="API request timeout in seconds",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Temperature for model responses (0.0 for reproducibility)",
    )

    # Rate limiting (NFR-7)
    rate_limit_requests_per_minute: int = Field(
        default=60,
        ge=1,
        description="Maximum API requests per minute per provider",
    )

    # Budget controls (NFR-9, NFR-10, NFR-11)
    max_cost_usd: float | None = Field(
        default=None,
        ge=0.0,
        description="Maximum total cost in USD (stops evaluation if exceeded)",
    )
    budget_warning_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Warn when this fraction of budget is reached",
    )

    # Checkpoint settings (FR-5)
    checkpoint_interval: int = Field(
        default=100,
        ge=10,
        description="Save checkpoint every N comparisons",
    )
    auto_resume: bool = Field(
        default=True,
        description="Automatically resume from last checkpoint if available",
    )

    # Paths
    data_dir: Path = Field(
        default=Path("data"),
        description="Directory containing benchmark data",
    )
    results_dir: Path = Field(
        default=Path("results"),
        description="Directory for evaluation results",
    )

    # Circuit breaker settings (FR-1.1d)
    circuit_breaker_failure_threshold: int = Field(
        default=5,
        ge=1,
        description="Open circuit after this many consecutive failures",
    )
    circuit_breaker_success_threshold: int = Field(
        default=3,
        ge=1,
        description="Close circuit after this many successes in half-open state",
    )
    circuit_breaker_timeout: int = Field(
        default=60,
        ge=10,
        description="Seconds before transitioning from open to half-open",
    )

    # Retry settings (FR-1.1a)
    max_retries: int = Field(
        default=5,
        ge=0,
        le=10,
        description="Maximum retry attempts for failed requests",
    )
    retry_base_delay: float = Field(
        default=1.0,
        ge=0.1,
        description="Base delay in seconds for exponential backoff",
    )

    @field_validator("data_dir", "results_dir", mode="before")
    @classmethod
    def ensure_path(cls, v: str | Path) -> Path:
        """Convert string paths to Path objects."""
        return Path(v) if isinstance(v, str) else v

    @property
    def block_a_size(self) -> int:
        """Number of Block A tasks for current benchmark version."""
        return BENCHMARK_SIZES[self.benchmark_version]["block_a"]

    @property
    def block_b_size(self) -> int:
        """Number of Block B tasks for current benchmark version."""
        return BENCHMARK_SIZES[self.benchmark_version]["block_b"]

    def get_retry_delays(self) -> list[float]:
        """Get exponential backoff delay sequence.

        Returns delays: 1s, 2s, 4s, 8s, 16s (for default settings).
        """
        return [self.retry_base_delay * (2**i) for i in range(self.max_retries)]


class JudgeCalibrationThresholds(BaseSettings):
    """Thresholds for judge calibration (Section 6.1)."""

    model_config = SettingsConfigDict(
        env_prefix="UKRQUALBENCH_JUDGE_",
        extra="ignore",
    )

    mc_accuracy: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Minimum MC agreement accuracy",
    )
    gec_f1: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Minimum GEC F1 score",
    )
    russism_f1: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Minimum russism detection F1",
    )
    max_false_positive_rate: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Maximum false positive rate",
    )
    pairwise_consistency: float = Field(
        default=0.90,
        ge=0.0,
        le=1.0,
        description="Minimum pairwise consistency with gold standard",
    )
    max_position_bias: float = Field(
        default=0.05,
        ge=0.0,
        le=0.5,
        description="Maximum position bias deviation from 50%",
    )
    max_length_bias: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description="Maximum length bias correlation |r|",
    )
    min_final_score: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Minimum weighted final calibration score",
    )

    # Calibration weights
    weight_mc: float = Field(default=0.20)
    weight_gec: float = Field(default=0.20)
    weight_russism: float = Field(default=0.25)
    weight_fp: float = Field(default=0.15)
    weight_pairwise: float = Field(default=0.20)


class DetectorPerformanceThresholds(BaseSettings):
    """Performance thresholds for detectors (NFR-8)."""

    model_config = SettingsConfigDict(
        env_prefix="UKRQUALBENCH_DETECTOR_",
        extra="ignore",
    )

    russism_max_ms_per_1k_tokens: float = Field(
        default=10.0,
        ge=1.0,
        description="Maximum ms to detect russisms per 1K tokens",
    )
    markers_max_ms_per_1k_tokens: float = Field(
        default=5.0,
        ge=1.0,
        description="Maximum ms to detect positive markers per 1K tokens",
    )
    anglicism_max_ms_per_1k_tokens: float = Field(
        default=10.0,
        ge=1.0,
        description="Maximum ms to detect anglicisms per 1K tokens",
    )
    fertility_max_ms_per_1k_tokens: float = Field(
        default=2.0,
        ge=0.5,
        description="Maximum ms to calculate fertility per 1K tokens",
    )


class ProviderConfig(BaseSettings):
    """Provider-specific configuration."""

    model_config = SettingsConfigDict(extra="ignore")

    name: str
    enabled: bool = True
    max_concurrent_requests: int = 10
    rate_limit_rpm: int = 60
    timeout_seconds: int = 60
    circuit_breaker_enabled: bool = True


# Badge thresholds (Section 5.4)
BADGE_THRESHOLDS: dict[str, dict[str, float]] = {
    "gold": {"min_elo": 1700, "max_russism_rate": 1.0},
    "silver": {"min_elo": 1550, "max_russism_rate": 3.0},
    "bronze": {"min_elo": 1400, "max_russism_rate": 5.0},
    "caution": {"max_russism_rate": 10.0},
    "not_recommended": {"max_elo": 1300, "max_russism_rate": 20.0},
}


def get_badge(
    elo_rating: float, russism_rate: float
) -> Literal["gold", "silver", "bronze", "caution", "not_recommended", "none"]:
    """Determine badge based on ELO rating and russism rate.

    Args:
        elo_rating: Model's ELO rating.
        russism_rate: Russisms per 1000 tokens.

    Returns:
        Badge name.
    """
    if russism_rate > 20.0 or elo_rating < 1300:
        return "not_recommended"
    if russism_rate > 10.0:
        return "caution"
    if elo_rating > 1700 and russism_rate < 1.0:
        return "gold"
    if elo_rating > 1550 and russism_rate < 3.0:
        return "silver"
    if elo_rating > 1400 and russism_rate < 5.0:
        return "bronze"
    return "none"
