"""UkrQualBench: Benchmark for evaluating Ukrainian language quality in LLMs.

UkrQualBench evaluates linguistic quality (nativeness) of Ukrainian language in LLMs,
not cognitive abilities. Key principles:
- Pairwise over Absolute: Compare models against each other
- Positive over Negative: Reward native markers, not just penalize errors
- Real over Synthetic: Real corpus data over synthetic
- Calibrated Judges: LLM judges calibrated against gold standard
"""

from ukrqualbench.core.config import BenchmarkVersion, Config

__version__ = "3.1.0"
__all__ = [
    "BenchmarkVersion",
    "Config",
    "__version__",
]
