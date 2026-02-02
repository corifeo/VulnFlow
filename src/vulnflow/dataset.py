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

    @classmethod
    def _from_records(cls, records: list[VulnRecord]) -> VulnDataset:
        """Convert list of VulnRecord to columnar storage.

        Args:
            records: List of VulnRecord instances

        Returns:
            VulnDataset with columnar storage

        Note:
            Complex fields (lists, nested dataclasses) stored as JSON strings.
            CVD timestamps are NOT stored here - EventInference adds them later.
        """
        import json
        from dataclasses import asdict, is_dataclass
        from typing import Any

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

        dataset = cls()
        if not records:
            return dataset

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
            # Scoring (JSON-encoded lists)
            "cvss_scores": np.array(
                [encode_nested(r.cvss_scores) for r in records], dtype=object
            ),
            "epss_scores": np.array(
                [encode_nested(r.epss_scores) for r in records], dtype=object
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
