"""
Extractors - Data source readers

Provides:
- Extractor: Protocol for data extraction
- NVDExtractor: NVD JSON file reader

Layer: Extract
Dependencies: schema
Used by: dataset
"""

from vulnflow.extractors.base import Extractor

__all__ = ["Extractor"]
