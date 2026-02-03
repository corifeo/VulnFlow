"""
Extractors - Data source readers

Provides:
- Extractor: Protocol for data extraction
- BaseExtractor: ABC for extractors with format validation
- NVDExtractor: NVD JSON file reader

Layer: Extract
Dependencies: schema
Used by: dataset
"""

from vulnflow.extractors.base import BaseExtractor, Extractor

__all__ = ["Extractor", "BaseExtractor"]
