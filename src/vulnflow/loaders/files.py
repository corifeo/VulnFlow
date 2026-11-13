"""
File Loaders - Parquet, JSON writers (lossless archive path)

Provides:
- ParquetLoader: Writes to Parquet files
- JSONLoader: Writes to JSON files

File loaders are the *archive path* — always lossless, no status filtering.

Layer: Load
Dependencies: dataset
Used by: pipeline, user code
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from vulnflow.loaders.base import BaseLoader
from vulnflow.registry import Registry

if TYPE_CHECKING:
    import pyarrow as pa

    from vulnflow.dataset import VulnDataset


_DATETIME_SUFFIXES = ("_at", "_timestamp", "_date", "_added", "_due_date")


def _coerce_to_datetime(v: object) -> datetime | None:
    """Coerce a value to datetime, handling strings and None."""
    if v is None:
        return None
    if isinstance(v, datetime):
        return v
    if isinstance(v, str):
        try:
            return datetime.fromisoformat(v)
        except (ValueError, TypeError):
            return None
    return None


def _is_datetime_column(col: str, sample: object) -> bool:
    """Detect if a column contains datetime data (by sample type or name suffix)."""
    if isinstance(sample, datetime):
        return True
    if isinstance(sample, str) and any(col.endswith(s) for s in _DATETIME_SUFFIXES):
        # String in a datetime-named column — check if it parses
        try:
            datetime.fromisoformat(sample)
            return True
        except (ValueError, TypeError):
            return False
    return False


def _normalize_for_arrow(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Normalize mixed-type object arrays for PyArrow compatibility.

    PyArrow cannot infer types from object arrays containing mixed types
    (e.g., datetime + None, float + None, datetime + str). This inspects
    each column and converts to homogeneous types that PyArrow can handle.
    """
    result: dict[str, np.ndarray] = {}
    for col, arr in data.items():
        if arr.dtype != object or len(arr) == 0:
            result[col] = arr
            continue

        # Sample first non-None value to detect type
        sample = None
        for v in arr:
            if v is not None:
                sample = v
                break

        if sample is None:
            # All None — keep as object (PyArrow infers null column)
            result[col] = arr
        elif _is_datetime_column(col, sample):
            # Coerce mixed datetime/string/None → PyArrow timestamp
            import pyarrow as pa

            values = [_coerce_to_datetime(v) for v in arr]
            result[col] = pa.array(values, type=pa.timestamp("us", tz="UTC"))
        elif isinstance(sample, float):
            # Convert float + None → float64 with NaN
            result[col] = np.array(
                [v if v is not None else np.nan for v in arr], dtype=np.float64
            )
        elif isinstance(sample, bool):
            # bool + None — keep as object (PyArrow handles bool/None fine)
            result[col] = arr
        else:
            result[col] = arr

    return result


def _utc_timestamps(table: pa.Table) -> pa.Table:
    """Label naive timestamp columns as UTC.

    numpy datetime64 has no timezone, so columns read back from Parquet are
    naive. VulnFlow stores every timestamp in UTC, so the values don't change.
    """
    import pyarrow as pa

    for i, field in enumerate(table.schema):
        if pa.types.is_timestamp(field.type) and field.type.tz is None:
            utc = pa.timestamp(field.type.unit, tz="UTC")
            table = table.set_column(i, field.with_type(utc), table.column(i).cast(utc))
    return table


@Registry.register_loader("parquet")
class ParquetLoader(BaseLoader):
    """Load dataset to Parquet file (lossless, no filtering)."""

    format: str = "parquet"
    name: str = "ParquetLoader"

    def __init__(self, path: str | Path) -> None:
        """Initialize loader with output path.

        Args:
            path: Output file path
        """
        super().__init__()
        self.path = Path(path)

    @property
    def label(self) -> str:
        """Short display name for progress/provenance."""
        return f"ParquetLoader({self.path.name})"

    def load(self, dataset: VulnDataset) -> dict[str, Any]:
        """Write dataset to Parquet file.

        Args:
            dataset: VulnDataset to write

        Returns:
            Metadata with path and record count
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        from vulnflow.utils import json as vjson

        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Normalize mixed-type object arrays for PyArrow compatibility
        normalized = _normalize_for_arrow(dataset._data)

        # Convert to PyArrow table
        table = _utc_timestamps(pa.table(normalized))

        # Write provenance to Parquet schema metadata
        provenance_bytes = vjson.dumps(dataset.provenance).encode("utf-8")
        existing_meta = table.schema.metadata or {}
        merged_meta = {**existing_meta, b"vulnflow:provenance": provenance_bytes}
        table = table.replace_schema_metadata(merged_meta)

        pq.write_table(table, self.path)

        return {"path": str(self.path), "record_count": len(dataset)}


@Registry.register_loader("json")
class JSONLoader(BaseLoader):
    """Load dataset to JSON file (lossless, no filtering)."""

    format: str = "json"
    name: str = "JSONLoader"

    def __init__(self, path: str | Path) -> None:
        """Initialize loader with output path.

        Args:
            path: Output file path
        """
        super().__init__()
        self.path = Path(path)

    @property
    def label(self) -> str:
        """Short display name for progress/provenance."""
        return f"JSONLoader({self.path.name})"

    def load(self, dataset: VulnDataset) -> dict[str, Any]:
        """Write dataset to JSON file.

        Args:
            dataset: VulnDataset to write

        Returns:
            Metadata with path and record count
        """
        from vulnflow.utils import json
        from vulnflow.utils.schema import serialize_value

        path = Path(self.path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert columnar data to list of dicts
        records = []
        if dataset._data:
            keys = list(dataset._data.keys())
            for i in range(len(dataset)):
                record = {}
                for k in keys:
                    value = dataset._data[k][i]
                    record[k] = serialize_value(value)
                records.append(record)

        with path.open("w", encoding="utf-8") as f:
            json.dump(records, f, indent=True)

        return {"path": str(self.path), "record_count": len(records)}
