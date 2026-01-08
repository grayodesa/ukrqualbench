"""Dataset source loaders for external datasets.

Each loader handles downloading, parsing, and converting external datasets
to UkrQualBench task schemas.
"""

from ukrqualbench.datasets.sources.brown_uk import BrownUKLoader
from ukrqualbench.datasets.sources.flores import FLORESLoader
from ukrqualbench.datasets.sources.ua_gec import UAGECLoader
from ukrqualbench.datasets.sources.zno import ZNOLoader

__all__ = [
    "BrownUKLoader",
    "FLORESLoader",
    "UAGECLoader",
    "ZNOLoader",
]
