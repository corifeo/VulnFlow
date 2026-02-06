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

import asyncio
import copy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from pandas import DataFrame as PandasDataFrame
from polars import DataFrame as PolarsDataFrame

from vulnflow.utils.graph import topological_sort
from vulnflow.utils.progress import ProgressReporter
from vulnflow.utils.transforms import partition_transforms


def _get_version() -> str:
    """Get vulnflow version without circular import."""
    from vulnflow import __version__

    return __version__


class VulnDataset:
    """Columnar storage for vulnerability records.

    Internal storage uses dict[str, np.ndarray] for vectorized operations.
    Public API provides extractors, transformers, and loaders.
    """

    def __init__(self) -> None:
        """Initialize empty dataset."""
        self._data: dict[str, np.ndarray] = {}
        self._provenance: dict[str, Any] = {
            "extractors": [],
            "transforms": [],
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
            "record_count": 0,
            "vulnflow_version": _get_version(),
        }

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

    @property
    def provenance(self) -> dict[str, Any]:
        """Return deep copy of provenance metadata.

        Returns:
            Dict with extractors, transforms, timestamps, record_count, vulnflow_version.
        """
        prov = copy.deepcopy(self._provenance)
        prov["record_count"] = len(self)
        return prov

    def _record_provenance(
        self,
        *,
        extractor: str | None = None,
        transform: str | None = None,
    ) -> None:
        """Record a provenance event (extractor or transform execution).

        Args:
            extractor: Extractor label to record (deduplicated).
            transform: Transform name to record (deduplicated).
        """
        if extractor and extractor not in self._provenance["extractors"]:
            self._provenance["extractors"].append(extractor)
        if transform and transform not in self._provenance["transforms"]:
            self._provenance["transforms"].append(transform)
        self._provenance["updated_at"] = datetime.now(UTC).isoformat()

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

    def insert_columns_after(self, after: str, columns: dict[str, np.ndarray]) -> None:
        """Insert new columns immediately after a specified column.

        Maintains column ordering for flatten transforms which should insert
        scalar columns right after their JSON parent column.

        Args:
            after: Column name to insert after
            columns: Dict of column_name -> array to insert

        Raises:
            KeyError: If 'after' column doesn't exist
            ValueError: If column length doesn't match dataset length

        Example:
            dataset.insert_columns_after("cwes", {
                "cwe_id": cwe_ids,
                "cwe_name": cwe_names
            })
        """
        if after not in self._data:
            msg = f"Column '{after}' not found in dataset"
            raise KeyError(msg)

        # Validate lengths
        for name, data in columns.items():
            if self._data and len(data) != len(self):
                msg = f"Column length mismatch: expected {len(self)}, got {len(data)}"
                raise ValueError(msg)

        # Rebuild dict with columns inserted after specified column
        new_data: dict[str, np.ndarray] = {}
        for key, value in self._data.items():
            new_data[key] = value
            if key == after:
                # Insert new columns here
                new_data.update(columns)

        self._data = new_data

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

        with ProgressReporter(
            "Transforming dataset",
            total=len(all_transforms),
            unit="transforms",
        ) as rpt:
            # Run async concurrently
            if async_transforms:

                async def run_all() -> None:
                    for t in async_transforms:
                        rpt.stage_started(t.name)  # type: ignore[attr-defined]
                    try:
                        await asyncio.gather(
                            *[t.transform(self) for t in async_transforms]  # type: ignore[attr-defined]
                        )
                        for t in async_transforms:
                            rpt.stage_completed(t.name)  # type: ignore[attr-defined]
                    except Exception as exc:
                        for t in async_transforms:
                            rpt.stage_failed(t.name, error=str(exc))  # type: ignore[attr-defined]
                        raise

                # Handle Jupyter/IPython event loops
                try:
                    asyncio.get_running_loop()
                    import nest_asyncio  # type: ignore[import-untyped]

                    nest_asyncio.apply()
                    asyncio.run(run_all())
                except RuntimeError:
                    asyncio.run(run_all())

            # Run sync sequentially
            for t in sync_transforms:
                t_name: str = t.name  # type: ignore[attr-defined]
                rpt.stage_started(t_name)
                t.transform(self)  # type: ignore[attr-defined]
                rpt.stage_completed(t_name)

        # Record provenance for all transforms
        for t in all_transforms:
            self._record_provenance(transform=t.name)  # type: ignore[attr-defined]

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

        # Merge provenance from all input datasets
        all_extractors: list[str] = []
        all_transforms: list[str] = []
        created_dates: list[str] = []

        for ds in datasets:
            for ext in ds._provenance["extractors"]:
                if ext not in all_extractors:
                    all_extractors.append(ext)
            for tfm in ds._provenance["transforms"]:
                if tfm not in all_transforms:
                    all_transforms.append(tfm)
            created_dates.append(ds._provenance["created_at"])

        merged._provenance["extractors"] = all_extractors
        merged._provenance["transforms"] = all_transforms
        merged._provenance["created_at"] = min(created_dates)

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

    def get_status_distribution(self) -> dict[str, int]:
        """Get distribution of nvd_status values.

        Returns:
            Dictionary mapping status values to counts

        Example:
            >>> ds = VulnDataset.from_nvd("data.json")
            >>> ds.get_status_distribution()
            {'analyzed': 1141, 'awaiting': 1506, 'modified': 89, 'rejected': 28, 'unknown': 3}
        """
        import numpy as np

        statuses = self.get_column("nvd_status")
        unique, counts = np.unique(statuses, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist(), strict=True))

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
