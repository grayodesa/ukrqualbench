"""Pytest configuration and fixtures for UkrQualBench tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from ukrqualbench.core.config import Config
from ukrqualbench.core.elo import ELOCalculator


@pytest.fixture
def config() -> Config:
    """Provide test configuration."""
    return Config(
        benchmark_version="lite",
        max_concurrent_requests=2,
        request_timeout=30,
    )


@pytest.fixture
def elo_calculator() -> ELOCalculator:
    """Provide fresh ELO calculator."""
    return ELOCalculator()


@pytest.fixture
def tmp_results_dir(tmp_path: Path) -> Path:
    """Provide temporary results directory."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    return results_dir
