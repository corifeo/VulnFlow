"""
File Loaders - Parquet, JSON, CSV writers

Provides:
- ParquetLoader: Writes to Parquet files
- JSONLoader: Writes to JSON files

Layer: Load
Dependencies: dataset
Used by: pipeline, user code
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from vulnflow.loaders.base import BaseLoader
from vulnflow.registry import Registry

if TYPE_CHECKING:
    from vulnflow.dataset import VulnDataset


@Registry.register_loader("parquet")
class ParquetLoader(BaseLoader):
    """Load dataset to Parquet file."""

    format: str = "parquet"
    name: str = "parquet"

    def __init__(self, path: str | Path) -> None:
        """Initialize loader with output path.

        Args:
            path: Output file path
        """
        super().__init__()
        self.path = Path(path)

    def load(self, dataset: VulnDataset) -> dict[str, Any]:
        """Write dataset to Parquet file.

        Args:
            dataset: VulnDataset to write

        Returns:
            Metadata with path and record count
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to PyArrow table and write
        table = pa.table(dataset._data)
        pq.write_table(table, self.path)

        return {"path": str(self.path), "record_count": len(dataset)}


@Registry.register_loader("json")
class JSONLoader(BaseLoader):
    """Load dataset to JSON file."""

    format: str = "json"
    name: str = "json"

    def __init__(self, path: str | Path) -> None:
        """Initialize loader with output path.

        Args:
            path: Output file path
        """
        super().__init__()
        self.path = Path(path)

    def load(self, dataset: VulnDataset) -> dict[str, Any]:
        """Write dataset to JSON file.

        Args:
            dataset: VulnDataset to write

        Returns:
            Metadata with path and record count
        """
        import json

        path = Path(self.path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert columnar data to list of dicts
        records = []
        if dataset._data:
            keys = list(dataset._data.keys())
            for i in range(len(dataset)):
                record = {k: dataset._data[k][i] for k in keys}
                # Convert non-serializable types
                for k, v in record.items():
                    if v is not None and not isinstance(v, (str, int, float, bool)):
                        record[k] = str(v)
                records.append(record)

        with path.open("w") as f:
            json.dump(records, f, indent=2, default=str)

        return {"path": str(self.path), "record_count": len(dataset)}
