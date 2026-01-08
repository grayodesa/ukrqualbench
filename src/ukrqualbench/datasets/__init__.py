"""Dataset loaders and benchmark assembly for UkrQualBench.

This module provides:
- Source loaders for external datasets (UA-GEC, ZNO, FLORES, Brown-UK)
- Benchmark loader for assembling complete benchmark datasets
- Data containers for Block A and Block B tasks
"""

from ukrqualbench.datasets.loader import (
    BENCHMARK_SPECS,
    BenchmarkData,
    BenchmarkLoader,
    BenchmarkMetadata,
    BlockAData,
    BlockBData,
)
from ukrqualbench.datasets.sources import (
    BrownUKLoader,
    FLORESLoader,
    UAGECLoader,
    ZNOLoader,
)

__all__ = [
    # Benchmark loader
    "BENCHMARK_SPECS",
    "BenchmarkData",
    "BenchmarkLoader",
    "BenchmarkMetadata",
    "BlockAData",
    "BlockBData",
    # Source loaders
    "BrownUKLoader",
    "FLORESLoader",
    "UAGECLoader",
    "ZNOLoader",
]
