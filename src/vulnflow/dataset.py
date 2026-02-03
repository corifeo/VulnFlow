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

from vulnflow.utils.graph import topological_sort

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl


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

    def get_column_or_none(self, name: str) -> np.ndarray | None:
        """Get column by name, returning None if not present.

        Args:
            name: Column name

        Returns:
            Column data as numpy array, or None if column doesn't exist
        """
        return self._data.get(name)

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

    def transform(self, transforms: list[object] | None = None) -> VulnDataset:
        """Apply transforms and enrichers to dataset (modifies in place).

        Handles both:
        - Sync transforms (with sync transform() method)
        - Async enrichers (with async transform() method) - run concurrently via asyncio.gather()

        Args:
            transforms: List of transform/enricher instances. If None, uses registered defaults
                       (both enrichers and transforms).

        Returns:
            Self for method chaining
        """
        import asyncio
        import inspect

        if transforms is None:
            from vulnflow.registry import Registry

            # Get both default enrichers and transforms
            enrichers = Registry.get_default_enrichers()
            computed_transforms = Registry.get_default_transforms()
            transforms = enrichers + computed_transforms

        # Sort by dependencies
        transforms = topological_sort(transforms)

        # Separate async from sync
        async_transforms = [
            t
            for t in transforms
            if hasattr(t, "transform") and inspect.iscoroutinefunction(t.transform)
        ]
        sync_transforms = [
            t
            for t in transforms
            if hasattr(t, "transform") and not inspect.iscoroutinefunction(t.transform)
        ]

        # Run async concurrently
        if async_transforms:
            async def run_all() -> None:
                await asyncio.gather(*[t.transform(self) for t in async_transforms])  # type: ignore[misc]

            asyncio.run(run_all())

        # Run sync sequentially
        for t in sync_transforms:
            t.transform(self)

        return self

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
        return extractor.extract()

    def to_pandas(self) -> pd.DataFrame:
        """Export dataset to pandas DataFrame.

        Returns:
            Pandas DataFrame with flattened scalar columns
        """
        from vulnflow.loaders.dataframe import PandasLoader

        loader = PandasLoader()
        return loader.load(self)

    def to_polars(self) -> pl.DataFrame:
        """Export dataset to polars DataFrame.

        Returns:
            Polars DataFrame with flattened scalar columns
        """
        from vulnflow.loaders.dataframe import PolarsLoader

        loader = PolarsLoader()
        return loader.load(self)

    def to_parquet(self, path: str | Path) -> None:
        """Export dataset to Parquet file.

        Args:
            path: Output file path
        """
        from vulnflow.loaders.files import ParquetLoader

        loader = ParquetLoader(path)
        loader.load(self)

    @classmethod
    def from_parquet(cls, path: str | Path) -> VulnDataset:
        """Load dataset from Parquet file.

        Args:
            path: Path to Parquet file

        Returns:
            VulnDataset with data from Parquet file
        """
        from vulnflow.extractors.files import ParquetExtractor

        extractor = ParquetExtractor(str(path))
        return extractor.extract()

    def update_sources(self, indices: list[int], source: str) -> None:
        """Update sources list for specified records.

        Args:
            indices: List of record indices to update
            source: Source name to add (e.g., "KEV", "EPSS")
        """
        import json
        from datetime import datetime

        # Make writable copies (Parquet load returns read-only arrays)
        sources_col = self.get_column("sources").copy()
        updated_at_col = self.get_column("last_updated_at").copy()

        now = datetime.now()

        for idx in indices:
            # Deserialize, update, reserialize
            sources_list = json.loads(sources_col[idx])
            if source not in sources_list:
                sources_list.append(source)
            sources_col[idx] = json.dumps(sources_list)

            # Update timestamp
            updated_at_col[idx] = now

        # Update columns with new arrays
        self._data["sources"] = sources_col
        self._data["last_updated_at"] = updated_at_col
