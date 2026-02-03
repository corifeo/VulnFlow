"""
File Extractors - Parquet, JSON, CSV readers

Provides:
- ParquetExtractor: Reads from Parquet files
- JSONExtractor: Reads from JSON files
- CSVExtractor: Reads from CSV files

Layer: Extract
Dependencies: None
Used by: dataset.from_parquet(), from_json(), from_csv()
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from vulnflow.extractors.base import BaseExtractor
from vulnflow.registry import Registry

if TYPE_CHECKING:
    from vulnflow.dataset import VulnDataset


@Registry.register_extractor("parquet")
class ParquetExtractor(BaseExtractor):
    """Extract from Parquet files (direct load).

    Parquet is already columnar, so we load directly to VulnDataset
    without intermediate VulnRecord conversion.
    """

    name: str = "parquet"
    format: str = "parquet"

    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = path

    def extract(self) -> VulnDataset:
        """Extract from Parquet → VulnDataset directly.

        Returns:
            VulnDataset with columnar data from Parquet file

        Raises:
            FileNotFoundError: If Parquet file doesn't exist
        """
        import pyarrow.parquet as pq

        from vulnflow.dataset import VulnDataset

        path = Path(self.path)
        if not path.exists():
            raise FileNotFoundError(f"Parquet file not found: {path}")

        table = pq.read_table(path)

        dataset = VulnDataset()
        dataset._data = {col: table[col].to_numpy() for col in table.column_names}

        return dataset
