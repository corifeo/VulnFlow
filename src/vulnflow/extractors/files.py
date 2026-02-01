"""
File Extractors - Parquet, JSON, CSV readers

Provides:
- ParquetExtractor: Reads from Parquet files
- JSONExtractor: Reads from JSON files
- CSVExtractor: Reads from CSV files

Layer: Extract
Dependencies: schema
Used by: dataset.from_parquet(), from_json(), from_csv()
"""

from __future__ import annotations

from vulnflow.extractors.base import Extractor
from vulnflow.schema import VulnRecord


class ParquetExtractor(Extractor):
    """Extracts vulnerability records from Parquet files."""

    def __init__(self, path: str) -> None:
        self.path = path

    def extract(self) -> list[VulnRecord]:
        """Extract vulnerability records from Parquet file."""
        raise NotImplementedError("Parquet extraction not yet implemented")
