"""
File Loaders - Parquet, JSON, CSV writers

Provides:
- ParquetLoader: Writes to Parquet files
- JSONLoader: Writes to JSON files
- CSVLoader: Writes to CSV files

Layer: Load
Dependencies: dataset, schema
Used by: dataset.to_parquet(), to_json(), to_csv()
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vulnflow.loaders.base import Loader

if TYPE_CHECKING:
    from vulnflow.dataset import VulnDataset


class ParquetLoader(Loader):
    """Writes vulnerability records to Parquet files."""

    def __init__(self, path: str) -> None:
        self.path = path

    def load(self, dataset: VulnDataset) -> None:
        """Write dataset to Parquet file."""
        raise NotImplementedError("Parquet loading not yet implemented")
