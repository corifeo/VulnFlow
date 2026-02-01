"""
NVD Extractor - NVD JSON file reader

Provides:
- NVDExtractor: Extracts vulnerability records from NVD JSON files

Layer: Extract
Dependencies: schema
Used by: dataset.from_nvd()
"""

from __future__ import annotations

from vulnflow.extractors.base import Extractor
from vulnflow.schema import VulnRecord


class NVDExtractor(Extractor):
    """Extracts vulnerability records from NVD JSON files."""

    def __init__(self, path: str) -> None:
        self.path = path

    def extract(self) -> list[VulnRecord]:
        """Extract vulnerability records from NVD JSON file(s)."""
        raise NotImplementedError("NVD extraction not yet implemented")
