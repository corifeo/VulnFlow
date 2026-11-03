"""
Extractors - Data source readers

Provides:
- Extractor: Protocol for data extraction
- BaseExtractor: ABC for extractors with format validation
- NVDExtractor: NVD JSON feed reader (files or URLs)
- NVDAPIExtractor: NVD REST API 2.0 client
- MSRCExtractor: Microsoft Patch Tuesday (CVRF API) reader
- ParquetExtractor, JSONExtractor: File readers
- PandasExtractor, PolarsExtractor: In-memory DataFrame readers

Layer: Extract
Dependencies: schema
Used by: dataset
"""

from vulnflow.extractors.base import BaseExtractor, Extractor
from vulnflow.extractors.files import (
    JSONExtractor,
    PandasExtractor,
    ParquetExtractor,
    PolarsExtractor,
)
from vulnflow.extractors.msrc import MSRCExtractor
from vulnflow.extractors.nvd import NVDExtractor
from vulnflow.extractors.nvd_api import NVDAPIExtractor

__all__ = [
    "Extractor",
    "BaseExtractor",
    "NVDExtractor",
    "NVDAPIExtractor",
    "MSRCExtractor",
    "ParquetExtractor",
    "JSONExtractor",
    "PandasExtractor",
    "PolarsExtractor",
]
