"""
VulnDataset - Core container for vulnerability records

Provides:
- VulnDataset: Main class for ETL operations on vulnerability data

Layer: Core
Dependencies: schema
Used by: __init__, pipeline, user code

Storage: Columnar (dict[str, np.ndarray]) for vectorized operations at scale
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

    from vulnflow.schema import VulnRecord


class VulnDataset:
    """Columnar storage for vulnerability records.

    Internal storage uses dict[str, np.ndarray] for vectorized operations.
    Public API provides extractors, transformers, and loaders.
    """

    def __init__(self) -> None:
        """Initialize empty dataset."""
        self._data: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        """Return number of records."""
        if not self._data:
            return 0
        # Get length from first column
        first_key = next(iter(self._data.keys()))
        return len(self._data[first_key])

    @property
    def columns(self) -> list[str]:
        """Return list of column names.

        Returns:
            List of column names in dataset
        """
        return list(self._data.keys())

    def get_column(self, name: str) -> np.ndarray:
        """Get column data as numpy array.

        Args:
            name: Column name

        Returns:
            Numpy array with column data

        Raises:
            KeyError: If column does not exist
        """
        return self._data[name]

    def set_column(self, name: str, data: np.ndarray) -> None:
        """Set column data (add new or update existing).

        Args:
            name: Column name
            data: Numpy array with column data

        Raises:
            ValueError: If data length doesn't match dataset length
        """
        if self._data and len(data) != len(self):
            msg = f"Column length mismatch: expected {len(self)}, got {len(data)}"
            raise ValueError(msg)

        self._data[name] = data

    def transform(self, transforms: list[object] | None = None) -> None:
        """Apply transforms to dataset (modifies in place).

        Args:
            transforms: List of transform instances. If None, uses registered defaults.
        """
        if transforms is None:
            from vulnflow.registry import get_transforms

            transform_classes = get_transforms()
            transforms = [cls() for cls in transform_classes]

        for t in transforms:
            if hasattr(t, "transform"):
                t.transform(self)

    @classmethod
    def _from_records(cls, records: list[VulnRecord]) -> VulnDataset:
        """Convert list of VulnRecord to columnar storage.

        Args:
            records: List of VulnRecord instances

        Returns:
            VulnDataset with columnar storage

        Note:
            Complex fields (lists, nested dataclasses) stored as JSON strings.
            Flattened columns (cvss_base_score, etc.) added for DataFrame convenience.
            CVD timestamps are NOT stored here - EventInference adds them later.
        """
        import json
        from dataclasses import asdict, is_dataclass
        from typing import Any

        from vulnflow.schema import CVSSEntry

        def encode_nested(value: Any) -> str:
            """Encode nested structure as JSON string."""
            if value is None:
                return json.dumps(None)
            if isinstance(value, list):
                items = []
                for item in value:
                    if is_dataclass(item) and not isinstance(item, type):
                        items.append(asdict(item))
                    else:
                        items.append(item)
                return json.dumps(items, default=str)
            if is_dataclass(value) and not isinstance(value, type):
                return json.dumps(asdict(value), default=str)
            return json.dumps(value, default=str)

        def flatten_cvss(
            cvss_scores: list[CVSSEntry],
        ) -> dict[str, float | str | None]:
            """Flatten list[CVSSEntry] to scalar columns.

            Selection logic:
            1. Prefer source: nvd > redhat > microsoft > other
            2. If tie, take first in list
            """
            if not cvss_scores:
                return {
                    "cvss_base_score": None,
                    "cvss_base_severity": None,
                    "cvss_version": None,
                    "cvss_vector_string": None,
                    "cvss_source": None,
                }

            # Source preference order
            source_priority = {"nvd": 0, "redhat": 1, "microsoft": 2}

            def score_key(entry: CVSSEntry) -> int:
                return source_priority.get(entry.source, 999)

            # Select preferred entry
            selected = min(cvss_scores, key=score_key)

            return {
                "cvss_base_score": selected.base_score,
                "cvss_base_severity": selected.base_severity,
                "cvss_version": selected.version,
                "cvss_vector_string": selected.vector_string,
                "cvss_source": selected.source,
            }

        dataset = cls()
        if not records:
            return dataset

        # Flatten CVSS for each record
        flattened_cvss = [flatten_cvss(r.cvss_scores) for r in records]

        dataset._data = {
            # Identity
            "id": np.array([str(r.id) for r in records], dtype=object),
            "cve_id": np.array([r.cve_id for r in records], dtype=object),
            "aliases": np.array([json.dumps(r.aliases) for r in records], dtype=object),
            # Description
            "title": np.array([r.title for r in records], dtype=object),
            "description": np.array([r.description for r in records], dtype=object),
            # Metadata
            "first_seen_at": np.array([r.first_seen_at for r in records], dtype=object),
            "last_updated_at": np.array(
                [r.last_updated_at for r in records], dtype=object
            ),
            "sources": np.array([json.dumps(r.sources) for r in records], dtype=object),
            # NVD Temporal
            "nvd_published_at": np.array(
                [r.nvd_published_at for r in records], dtype=object
            ),
            "nvd_modified_at": np.array(
                [r.nvd_modified_at for r in records], dtype=object
            ),
            # Scoring (JSON-encoded lists for Parquet round-trip)
            "cvss_scores": np.array(
                [encode_nested(r.cvss_scores) for r in records], dtype=object
            ),
            "epss_scores": np.array(
                [encode_nested(r.epss_scores) for r in records], dtype=object
            ),
            # Flattened CVSS (for DataFrame convenience)
            "cvss_base_score": np.array(
                [f["cvss_base_score"] for f in flattened_cvss], dtype=object
            ),
            "cvss_base_severity": np.array(
                [f["cvss_base_severity"] for f in flattened_cvss], dtype=object
            ),
            "cvss_version": np.array(
                [f["cvss_version"] for f in flattened_cvss], dtype=object
            ),
            "cvss_vector_string": np.array(
                [f["cvss_vector_string"] for f in flattened_cvss], dtype=object
            ),
            "cvss_source": np.array(
                [f["cvss_source"] for f in flattened_cvss], dtype=object
            ),
            # Exploitation (JSON-encoded)
            "kev": np.array([encode_nested(r.kev) for r in records], dtype=object),
            "exploits": np.array(
                [encode_nested(r.exploits) for r in records], dtype=object
            ),
            # Classification (JSON-encoded lists)
            "cwes": np.array([encode_nested(r.cwes) for r in records], dtype=object),
            "cpes": np.array([encode_nested(r.cpes) for r in records], dtype=object),
            # References (JSON-encoded list)
            "references": np.array(
                [encode_nested(r.references) for r in records], dtype=object
            ),
            # Extensible (JSON-encoded dicts)
            "tags": np.array([json.dumps(r.tags) for r in records], dtype=object),
            "metadata": np.array(
                [json.dumps(r.metadata, default=str) for r in records], dtype=object
            ),
        }

        return dataset

    @classmethod
    def from_nvd(cls, path: str | Path) -> VulnDataset:
        """Load dataset from NVD JSON file.

        Args:
            path: Path to NVD JSON file

        Returns:
            VulnDataset with records from NVD file
        """
        from vulnflow.extractors.nvd import NVDExtractor

        extractor = NVDExtractor(str(path))
        records = extractor.extract()
        return cls._from_records(records)

    def to_pandas(self) -> pd.DataFrame:
        """Export dataset to pandas DataFrame.

        Returns:
            Pandas DataFrame with all columns
        """
        import pandas as pd

        return pd.DataFrame(self._data)

    def to_polars(self) -> pl.DataFrame:
        """Export dataset to polars DataFrame.

        Returns:
            Polars DataFrame with all columns
        """
        import polars as pl

        # Convert numpy arrays to lists for polars compatibility
        data = {k: v.tolist() for k, v in self._data.items()}
        return pl.from_dict(data)

    def to_parquet(self, path: str | Path) -> None:
        """Export dataset to Parquet file.

        Args:
            path: Output file path
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to PyArrow table
        table = pa.table(self._data)
        pq.write_table(table, path)

    @classmethod
    def from_parquet(cls, path: str | Path) -> VulnDataset:
        """Load dataset from Parquet file.

        Args:
            path: Path to Parquet file

        Returns:
            VulnDataset with data from Parquet file
        """
        import pyarrow.parquet as pq

        table = pq.read_table(path)

        dataset = cls()
        # Convert PyArrow table to dict of numpy arrays
        dataset._data = {col: table[col].to_numpy() for col in table.column_names}

        return dataset
