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

import numpy as np
from pandas import DataFrame as PandasDataFrame
from polars import DataFrame as PolarsDataFrame

from vulnflow.utils.graph import topological_sort


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

        Shows progress bar during execution (auto-detects Jupyter vs terminal).

        Args:
            transforms: List of transform/enricher instances. If None, uses registered defaults
                       (both enrichers and transforms).

        Returns:
            Self for method chaining
        """
        import asyncio

        from vulnflow.utils.progress import progress_bar
        from vulnflow.utils.transforms import partition_transforms

        if transforms is None:
            from vulnflow.registry import Registry

            # Get all default transforms (enrichers, inferences, computed)
            transforms = Registry.get_default_transforms()

        # Sort by dependencies
        transforms = topological_sort(transforms)

        # Separate async from sync
        async_transforms, sync_transforms = partition_transforms(transforms)

        # All transforms for progress bar
        all_transforms = async_transforms + sync_transforms

        with progress_bar(
            total=len(all_transforms),
            desc="Transforming dataset",
            unit="transforms",
        ) as pbar:
            # Run async concurrently (NO individual progress updates)
            if async_transforms:

                async def run_all() -> None:
                    await asyncio.gather(
                        *[t.transform(self) for t in async_transforms]  # type: ignore[attr-defined]
                    )

                asyncio.run(run_all())
                # Update progress AFTER all async complete
                pbar.update(len(async_transforms))

            # Run sync sequentially (WITH individual progress updates)
            for t in sync_transforms:
                pbar.set_postfix_str(f"[{t.name}]")  # type: ignore[attr-defined]
                t.transform(self)  # type: ignore[attr-defined]
                pbar.update(1)

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

    @classmethod
    def merge(cls, datasets: list[VulnDataset]) -> VulnDataset:
        """Merge multiple datasets into one.

        Concatenates columnar data from multiple datasets. Handles missing
        columns by filling with None.

        Args:
            datasets: List of VulnDataset instances to merge

        Returns:
            Merged VulnDataset with concatenated columns. Returns empty dataset
            if input list is empty, or first dataset unchanged if only one input.
        """
        if not datasets:
            return cls()

        if len(datasets) == 1:
            return datasets[0]

        # Get all columns (union)
        all_columns: set[str] = set()
        for ds in datasets:
            all_columns.update(ds.columns)

        # Concatenate arrays
        merged = cls()

        for col in all_columns:
            arrays: list[np.ndarray] = []
            for ds in datasets:
                # Get column or create None-filled array if missing
                col_data = ds.get_column_or_none(col)
                if col_data is not None:
                    arrays.append(col_data)
                else:
                    # Fill with None for missing column
                    arrays.append(np.full(len(ds), None, dtype=object))

            merged.set_column(col, np.concatenate(arrays))

        return merged

    def to_pandas(self, include_nested: bool = False) -> PandasDataFrame:
        """Export dataset to pandas DataFrame.

        Args:
            include_nested: If True, include nested JSON columns (cvss_scores, etc.)
                           for full data fidelity. Default False excludes nested columns.

        Returns:
            Pandas DataFrame
        """
        from vulnflow.loaders.dataframe import PandasLoader

        loader = PandasLoader(include_nested=include_nested)
        return loader.load(self)

    def to_polars(self, include_nested: bool = False) -> PolarsDataFrame:
        """Export dataset to polars DataFrame.

        Args:
            include_nested: If True, include nested JSON columns (cvss_scores, etc.)
                           for full data fidelity. Default False excludes nested columns.

        Returns:
            Polars DataFrame
        """
        from vulnflow.loaders.dataframe import PolarsLoader

        loader = PolarsLoader(include_nested=include_nested)
        return loader.load(self)

    def to_parquet(self, path: str | Path) -> None:
        """Export dataset to Parquet file.

        Args:
            path: Output file path
        """
        from vulnflow.loaders.files import ParquetLoader

        loader = ParquetLoader(path)
        loader.load(self)

    def to_json(self, path: str | Path) -> None:
        """Export dataset to JSON file.

        Args:
            path: Output file path
        """
        from vulnflow.loaders.files import JSONLoader

        loader = JSONLoader(path)
        loader.load(self)

    @classmethod
    def from_parquet(cls, path: str | list[str]) -> VulnDataset:
        """Load from Parquet file(s) → VulnDataset.

        Supports:
        - Single file: `VulnDataset.from_parquet("data.parquet")`
        - Multiple files with upsert: `VulnDataset.from_parquet(["cache.parquet", "delta.parquet"])`

        Multi-file upsert semantics:
        - Files loaded in order
        - Duplicate cve_ids: later files win
        - Schema mismatch: missing columns filled with None

        Args:
            path: File path or list of paths

        Returns:
            VulnDataset loaded from Parquet

        Raises:
            FileNotFoundError: If file doesn't exist
            ExtractorError: If multi-file merge attempted without cve_id

        Example:
            >>> # Single file
            >>> dataset = VulnDataset.from_parquet("data.parquet")
            >>>
            >>> # Multi-file with delta update
            >>> dataset = VulnDataset.from_parquet(["cache.parquet", "delta.parquet"])
        """
        from vulnflow.extractors.files import ParquetExtractor

        extractor = ParquetExtractor(path)
        return extractor.extract()

    @classmethod
    def from_json(cls, path: str | Path) -> VulnDataset:
        """Create dataset from JSON file (schema-aware round-trip).

        Uses Registry schema for accurate type reconstruction (datetime, UUID).
        For lossless round-trip with full type fidelity, use from_parquet().

        Args:
            path: Path to JSON file

        Returns:
            VulnDataset with data from JSON file
        """
        from vulnflow.extractors.files import JSONExtractor

        extractor = JSONExtractor(str(path))
        return extractor.extract()

    @classmethod
    def from_pandas(cls, dataframe: PandasDataFrame) -> VulnDataset:
        """Create dataset from pandas DataFrame (in-memory).

        Args:
            dataframe: Pandas DataFrame with flattened columns

        Returns:
            VulnDataset with data from DataFrame
        """
        from vulnflow.extractors.files import PandasExtractor

        extractor = PandasExtractor(dataframe)
        return extractor.extract()

    @classmethod
    def from_polars(cls, dataframe: PolarsDataFrame) -> VulnDataset:
        """Create dataset from Polars DataFrame (in-memory).

        Args:
            dataframe: Polars DataFrame with flattened columns

        Returns:
            VulnDataset with data from DataFrame
        """
        from vulnflow.extractors.files import PolarsExtractor

        extractor = PolarsExtractor(dataframe)
        return extractor.extract()

    def update_sources(self, indices: list[int], source: str) -> None:
        """Update sources list for specified records.

        Args:
            indices: List of record indices to update
            source: Source name to add (e.g., "KEV", "EPSS")
        """
        from datetime import datetime

        from vulnflow.utils import json

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
