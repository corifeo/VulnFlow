"""
File Extractors - Parquet, JSON, CSV readers

Provides:
- ParquetExtractor: Reads from Parquet files
- JSONExtractor: Reads from JSON files (schema-aware round-trip)
- PandasExtractor: Reads from pandas DataFrames
- PolarsExtractor: Reads from Polars DataFrames

Layer: Extract
Dependencies: None
Used by: dataset.from_parquet(), from_json(), from_pandas(), from_polars()
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from vulnflow.extractors.base import BaseExtractor
from vulnflow.registry import Registry

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

    from vulnflow.dataset import VulnDataset


@Registry.register_extractor("parquet")
class ParquetExtractor(BaseExtractor):
    """Extract from Parquet files (direct load with optional multi-file merge).

    Parquet is already columnar, so we load directly to VulnDataset
    without intermediate VulnRecord conversion.

    Features:
    - Single file: Fast direct load
    - Multiple files: Automatic upsert merge (later files win on cve_id conflicts)
    - Schema union: Missing columns filled with None

    Use cases:
    - Single file: `ParquetExtractor("data.parquet")`
    - Incremental: `ParquetExtractor(["cache.parquet", "delta.parquet"])`

    Upsert semantics for multi-file:
    1. Load files in order
    2. Concatenate all records
    3. Deduplicate by cve_id, keeping last occurrence
    4. Result: Latest version of each CVE

    Example:
        >>> # Incremental workflow
        >>> cache = "nvd_cache.parquet"  # 300K CVEs
        >>> delta = "nvd_delta.parquet"  # 2K modified/new CVEs
        >>> extractor = ParquetExtractor([cache, delta])
        >>> dataset = extractor.extract()  # Merged with delta overwrites
    """

    name: str = "parquet"
    format: str = "parquet"

    def __init__(self, path: str | list[str]) -> None:
        """Initialize Parquet extractor.

        Args:
            path: Single file path or list of paths for upsert merge
        """
        super().__init__()
        self.path = path

    def extract(self) -> VulnDataset:
        """Extract from Parquet → VulnDataset directly.

        Supports:
        - Single file: Direct load
        - Multiple files: Load all, merge with upsert (later files win)

        Returns:
            VulnDataset with columnar data from Parquet file(s)

        Raises:
            FileNotFoundError: If any Parquet file doesn't exist
            ExtractorError: If multi-file merge attempted without cve_id column
        """
        from vulnflow.dataset import VulnDataset

        # Handle empty list
        if isinstance(self.path, list) and len(self.path) == 0:
            return VulnDataset()

        # Normalize to list (handle str, Path, or list)
        if isinstance(self.path, str):
            paths = [self.path]
        elif isinstance(self.path, Path):
            paths = [str(self.path)]
        else:
            paths = self.path

        # Fast path: single file
        if len(paths) == 1:
            return self._load_single(Path(paths[0]))

        # Multi-file: load all then merge with upsert
        datasets = []
        for path_str in paths:
            datasets.append(self._load_single(Path(path_str)))

        return self._merge_upsert(datasets)

    def _load_single(self, path: Path) -> VulnDataset:
        """Load single Parquet file.

        Args:
            path: Path to parquet file

        Returns:
            VulnDataset with file contents

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        import pyarrow.parquet as pq

        from vulnflow.dataset import VulnDataset

        if not path.exists():
            from vulnflow.exceptions import ExtractorError

            raise ExtractorError(f"Parquet file not found: {path}")

        table = pq.read_table(path)

        dataset = VulnDataset()
        dataset._data = {col: table[col].to_numpy() for col in table.column_names}

        # Restore provenance from Parquet metadata if present
        meta = table.schema.metadata or {}
        prov_bytes = meta.get(b"vulnflow:provenance")
        if prov_bytes:
            from vulnflow.utils import json as vjson

            stored_prov = vjson.loads(prov_bytes)
            dataset._provenance["extractors"] = stored_prov.get("extractors", [])
            dataset._provenance["transforms"] = stored_prov.get("transforms", [])
            dataset._provenance["created_at"] = stored_prov.get(
                "created_at", dataset._provenance["created_at"]
            )
            if "date_min" in stored_prov:
                dataset._provenance["date_min"] = stored_prov["date_min"]
            if "date_max" in stored_prov:
                dataset._provenance["date_max"] = stored_prov["date_max"]

        # Record this extraction in provenance
        dataset._record_provenance(extractor=self.provenance_label)
        return dataset

    def _merge_upsert(self, datasets: list[VulnDataset]) -> VulnDataset:
        """Merge multiple datasets with upsert semantics (later wins).

        Algorithm:
        1. Concatenate all columns from all datasets
        2. Find duplicates by cve_id
        3. Keep last occurrence (later datasets override earlier)

        Args:
            datasets: List of datasets to merge

        Returns:
            Merged dataset with unique cve_ids

        Raises:
            ExtractorError: If cve_id column missing (required for upsert)
        """
        from vulnflow.dataset import VulnDataset
        from vulnflow.exceptions import ExtractorError

        if not datasets:
            return VulnDataset()

        # Verify all datasets have cve_id column
        for ds in datasets:
            if "cve_id" not in ds.columns:
                msg = "cve_id column required for multi-file merge"
                raise ExtractorError(msg)

        # Get union of all columns across datasets - preserve insertion order
        all_columns: dict[str, None] = {}
        for ds in datasets:
            for col in ds.columns:
                all_columns[col] = None

        # Concatenate all datasets, filling missing columns with None
        merged_data: dict[str, np.ndarray] = {}

        for col in all_columns.keys():
            arrays = []
            for ds in datasets:
                if col in ds.columns:
                    arrays.append(ds.get_column(col))
                else:
                    # Fill missing column with None
                    none_array = np.full(len(ds), None, dtype=object)
                    arrays.append(none_array)

            merged_data[col] = np.concatenate(arrays)

        # Find unique CVE IDs, keeping last occurrence
        cve_ids = merged_data["cve_id"]

        # Use dict to track last index for each CVE ID
        last_indices = {}
        for idx, cve_id in enumerate(cve_ids):
            last_indices[cve_id] = idx

        # Get unique indices (numpy fancy indexing handles unsorted)
        unique_indices = np.fromiter(
            last_indices.values(), dtype=np.intp, count=len(last_indices)
        )

        # Filter all columns to unique indices
        result = VulnDataset()
        for col, array in merged_data.items():
            result._data[col] = array[unique_indices]

        # Merge provenance from all input datasets

        all_extractors: list[str] = []
        all_transforms: list[str] = []
        created_dates: list[str] = []

        for ds in datasets:
            for ext in ds._provenance.get("extractors", []):
                if ext not in all_extractors:
                    all_extractors.append(ext)
            for tfm in ds._provenance.get("transforms", []):
                if tfm not in all_transforms:
                    all_transforms.append(tfm)
            created_dates.append(ds._provenance.get("created_at", ""))

        result._provenance["extractors"] = all_extractors
        result._provenance["transforms"] = all_transforms
        if created_dates:
            result._provenance["created_at"] = min(d for d in created_dates if d)

        # Merge date ranges
        date_mins = [
            ds._provenance["date_min"]
            for ds in datasets
            if "date_min" in ds._provenance
        ]
        date_maxes = [
            ds._provenance["date_max"]
            for ds in datasets
            if "date_max" in ds._provenance
        ]
        if date_mins:
            result._provenance["date_min"] = min(date_mins)
        if date_maxes:
            result._provenance["date_max"] = max(date_maxes)

        return result


@Registry.register_extractor("json")
class JSONExtractor(BaseExtractor):
    """Extract from JSON files (schema-aware round-trip from JSONLoader).

    Uses Registry schema to reconstruct accurate types:
    - datetime, UUID: Proper reconstruction (not lossy!)
    - int, float, bool: Typed conversion
    - JSON-encoded columns: Preserved as strings

    For maximum performance, use ParquetExtractor.
    """

    name: str = "json"
    format: str = "json"

    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = path

    def extract(self) -> VulnDataset:
        """Extract from JSON → VulnDataset with schema-aware type reconstruction.

        Returns:
            VulnDataset with schema-aware type conversion

        Raises:
            FileNotFoundError: If JSON file doesn't exist
            ValueError: If JSON is not a list
        """
        from vulnflow.dataset import VulnDataset
        from vulnflow.utils.schema import build_type_map, deserialize_value

        path = Path(self.path)
        self._ensure_file_exists(path)

        data = self._read_json_safely(path)

        if not isinstance(data, list):
            from vulnflow.exceptions import ExtractorError

            raise ExtractorError(f"Expected JSON array, got {type(data).__name__}")

        if not data:
            return VulnDataset()

        # Build type map once from Registry
        type_map = build_type_map()

        # Extract all column names - preserve insertion order
        all_columns: dict[str, None] = {}
        for record in data:
            for col in record.keys():
                all_columns[col] = None

        # Build columnar data
        columnar_data: dict[str, np.ndarray] = {}

        for col in all_columns.keys():
            # Collect all values for this column
            values = []
            for record in data:
                value = record.get(col)
                converted = deserialize_value(value, col, type_map)
                values.append(converted)

            # Create numpy array with object dtype to handle heterogeneous types
            columnar_data[col] = np.array(values, dtype=object)

        dataset = VulnDataset()
        dataset._data = columnar_data

        # Warn if core fields missing
        if "cve_id" not in all_columns:
            self.logger.warning("JSON dataset missing 'cve_id' column")
        if "description" not in all_columns:
            self.logger.warning("JSON dataset missing 'description' column")

        dataset._record_provenance(extractor=self.provenance_label)
        return dataset


@Registry.register_extractor("pandas")
class PandasExtractor(BaseExtractor):
    """Extract from pandas DataFrame (in-memory).

    Accepts flattened DataFrames (scalar columns only).
    Does NOT reconstruct nested structures.
    """

    name: str = "pandas"
    format: str = "pandas"

    def __init__(self, dataframe: pd.DataFrame) -> None:
        super().__init__()
        self.dataframe = dataframe

    def extract(self) -> VulnDataset:
        """Extract from DataFrame → VulnDataset.

        Returns:
            VulnDataset with columnar data from DataFrame
        """
        from vulnflow.dataset import VulnDataset

        if self.dataframe.empty:
            return VulnDataset()

        # Convert columns to numpy arrays
        dataset = VulnDataset()
        dataset._data = {
            col: self.dataframe[col].to_numpy() for col in self.dataframe.columns
        }

        # Warn if core fields missing
        if "cve_id" not in self.dataframe.columns:
            self.logger.warning("DataFrame missing 'cve_id' column")
        if "description" not in self.dataframe.columns:
            self.logger.warning("DataFrame missing 'description' column")

        dataset._record_provenance(extractor=self.provenance_label)
        return dataset


@Registry.register_extractor("polars")
class PolarsExtractor(BaseExtractor):
    """Extract from Polars DataFrame (direct column conversion)."""

    name: str = "polars"
    format: str = "polars"

    def __init__(self, dataframe: pl.DataFrame) -> None:
        super().__init__()
        self.dataframe = dataframe

    def extract(self) -> VulnDataset:
        """Extract from Polars → VulnDataset directly.

        Converts each column via Polars .to_numpy() without
        intermediate Pandas DataFrame.

        Returns:
            VulnDataset with columnar data from Polars DataFrame
        """
        from vulnflow.dataset import VulnDataset

        if self.dataframe.is_empty():
            return VulnDataset()

        dataset = VulnDataset()
        dataset._data = {
            col: self.dataframe[col].to_numpy() for col in self.dataframe.columns
        }

        # Warn if core fields missing
        if "cve_id" not in self.dataframe.columns:
            self.logger.warning("Polars DataFrame missing 'cve_id' column")
        if "description" not in self.dataframe.columns:
            self.logger.warning("Polars DataFrame missing 'description' column")

        dataset._record_provenance(extractor=self.provenance_label)
        return dataset
