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
from typing import TYPE_CHECKING, Any

import numpy as np

from vulnflow.utils.merge import dedupe_columns
from vulnflow.utils.progress import ProgressReporter

if TYPE_CHECKING:
    from pandas import DataFrame as PandasDataFrame
    from polars import DataFrame as PolarsDataFrame


class VulnDataset:
    """Core container for vulnerability records with columnar storage.

    Thin wrapper around ``dict[str, np.ndarray]`` that serves as the single
    data object flowing through the ETL pipeline. Delegates extraction to
    extractors, enrichment/computation to transforms, and output to loaders.

    Construction:
        VulnDataset.from_nvd("cve-2024.json")          # NVD JSON file
        VulnDataset.from_parquet("library.parquet")     # Parquet (lossless)
        VulnDataset.from_parquet(["base.parquet", "delta.parquet"])  # upsert
        VulnDataset.from_json("export.json")            # JSON round-trip
        VulnDataset.from_pandas(df)                     # pandas DataFrame
        VulnDataset.from_polars(df)                     # polars DataFrame

    Transform (modifies in place, returns self for chaining):
        ds.transform()                     # run registered defaults
        ds.transform([KEVEnricher(), ...])  # explicit transform list

    Export — analysis path (filters rejected rows by default):
        ds.to_pandas()                     # pandas DataFrame
        ds.to_polars()                     # polars DataFrame
        ds.to_pandas(rejected=True)        # include all statuses
        ds.to_pandas(start="2024-01-01")   # date range filter

    Export — archive path (lossless, no filtering):
        ds.to_parquet("out.parquet")
        ds.to_json("out.json")

    Query:
        ds.count_by("month")               # temporal distribution
        ds.get_status_distribution()        # NVD status counts
        ds.filter(mask)                     # boolean indexing → new dataset
        VulnDataset.merge([ds1, ds2])       # concatenate datasets
    """

    def __init__(self) -> None:
        """Initialize empty dataset."""
        from vulnflow import __version__

        self._data: dict[str, np.ndarray] = {}
        self._provenance: dict[str, Any] = {
            "extractors": [],
            "transforms": [],
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
            "record_count": 0,
            "vulnflow_version": __version__,
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
        for name, data in columns.items():  # noqa: B007
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

    def transform(
        self,
        transforms: list[object] | None = None,
        *,
        quiet: bool = False,
    ) -> VulnDataset:
        """Apply transforms and enrichers to dataset (modifies in place).

        Uses dependency-aware phased execution: transforms are grouped into
        phases by build_phases(). Within each phase, async transforms run
        via gather() and sync transforms run via to_thread(). A barrier
        separates each phase.

        Shows progress bar during execution (auto-detects Jupyter vs terminal).

        Args:
            transforms: List of transform/enricher instances. If None, uses registered defaults
                       (both enrichers and transforms).
            quiet: Suppress all progress output (for programmatic use).

        Returns:
            Self for method chaining
        """
        from vulnflow.executor import PipelineExecutor
        from vulnflow.utils.progress import _suppress_progress

        suppress_token = None
        if quiet:
            suppress_token = _suppress_progress.set(True)

        try:
            if transforms is None:
                from vulnflow.registry import Registry

                transforms = Registry.get_default_transforms()

            if not transforms:
                return self

            with ProgressReporter(
                "Enriching & transforming",
                total=len(transforms),
                unit="transforms",
            ) as rpt:
                # Apply nest_asyncio if already inside an event loop (Jupyter)
                try:
                    asyncio.get_running_loop()
                    import nest_asyncio  # type: ignore[import-untyped]

                    nest_asyncio.apply()
                except RuntimeError:
                    pass

                successful, _failed, _errors = asyncio.run(
                    PipelineExecutor.run_transforms(
                        self, transforms, strict=True, reporter=rpt
                    )
                )

            # Record provenance for successful transforms
            for t_name in successful:
                self._record_provenance(transform=t_name)

            return self
        finally:
            if suppress_token is not None:
                _suppress_progress.reset(suppress_token)

    @classmethod
    def from_nvd(cls, path: str | Path) -> VulnDataset:
        """Load dataset from NVD JSON file.

        Args:
            path: Path to NVD JSON file or URL

        Returns:
            VulnDataset with records from NVD file

        Raises:
            ExtractorError: On any failure (missing file, HTTP 404,
                bad format, corrupt archive, invalid JSON)
        """
        from vulnflow.exceptions import ExtractorError
        from vulnflow.extractors.nvd import NVDExtractor

        try:
            extractor = NVDExtractor(str(path))
            return extractor.extract()
        except ExtractorError as e:
            raise ExtractorError(str(e)) from None
        except Exception as e:
            raise ExtractorError(f"Failed to load NVD data from {path}: {e}") from None

    @classmethod
    def merge(cls, datasets: list[VulnDataset]) -> VulnDataset:
        """Merge multiple datasets into one.

        Concatenates columnar data from multiple datasets. Handles missing
        columns by filling with None. Keeps one row per ``cve_id``: the row
        with the newest ``nvd_modified_at`` wins, and ties or missing dates
        go to the later dataset. Datasets without ``cve_id`` are only
        concatenated.

        Args:
            datasets: List of VulnDataset instances to merge

        Returns:
            Merged VulnDataset. Returns empty dataset if input list is empty,
            or first dataset unchanged if only one input.
        """
        if not datasets:
            return cls()

        if len(datasets) == 1:
            return datasets[0]

        # Get all columns (union) - preserve insertion order
        all_columns: dict[str, None] = {}
        for ds in datasets:
            for col in ds.columns:
                all_columns[col] = None

        # Concatenate arrays
        merged = cls()
        concatenated: dict[str, np.ndarray] = {}

        for col in all_columns.keys():
            arrays: list[np.ndarray] = []
            for ds in datasets:
                # Get column or create None-filled array if missing
                col_data = ds.get_column_or_none(col)
                if col_data is not None:
                    arrays.append(col_data)
                else:
                    # Fill with None for missing column
                    arrays.append(np.full(len(ds), None, dtype=object))

            concatenated[col] = np.concatenate(arrays)

        for col, values in dedupe_columns(concatenated).items():
            merged.set_column(col, values)

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

        # Merge date ranges (min of mins, max of maxes)
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
            merged._provenance["date_min"] = min(date_mins)
        if date_maxes:
            merged._provenance["date_max"] = max(date_maxes)

        return merged

    def filter(self, mask: np.ndarray) -> VulnDataset:
        """Return new dataset containing only rows where mask is True.

        Args:
            mask: Boolean numpy array, same length as dataset

        Returns:
            New VulnDataset with selected rows and copied provenance

        Raises:
            TypeError: If mask is not a numpy ndarray
            ValueError: If mask dtype is not bool or length doesn't match
        """
        if not isinstance(mask, np.ndarray):
            msg = f"mask must be a numpy ndarray, got {type(mask).__name__}"
            raise TypeError(msg)
        if mask.dtype != np.bool_:
            msg = f"mask must be boolean dtype, got {mask.dtype}"
            raise ValueError(msg)
        if len(mask) != len(self):
            msg = f"mask length mismatch: expected {len(self)}, got {len(mask)}"
            raise ValueError(msg)

        result = VulnDataset()
        for col_name, col_data in self._data.items():
            result._data[col_name] = col_data[mask]
        result._provenance = copy.deepcopy(self._provenance)

        # Recompute date range from filtered nvd_published_at
        if "nvd_published_at" in result._data:
            pub_dates = result._data["nvd_published_at"]
            non_null = [d for d in pub_dates if d is not None]
            if non_null:
                date_min = min(non_null)
                date_max = max(non_null)
                result._provenance["date_min"] = (
                    date_min.isoformat()
                    if hasattr(date_min, "isoformat")
                    else str(date_min)
                )
                result._provenance["date_max"] = (
                    date_max.isoformat()
                    if hasattr(date_max, "isoformat")
                    else str(date_max)
                )
            else:
                result._provenance.pop("date_min", None)
                result._provenance.pop("date_max", None)

        return result

    def count_by(
        self,
        period: str = "month",
        *,
        start: str | datetime | None = None,
        end: str | datetime | None = None,
        rejected: bool = False,
    ) -> PandasDataFrame:
        """Count records grouped by time period.

        Args:
            period: Grouping period — 'year', 'quarter', 'month', 'week', or 'day'.
            start: Inclusive lower bound (>=) for nvd_published_at.
            end: Exclusive upper bound (<) for nvd_published_at.
            rejected: Include rejected rows (default False).

        Returns:
            Pandas DataFrame with columns [period, 'count'], sorted chronologically.

        Raises:
            ValueError: If period is not a valid option.
        """
        from vulnflow.utils.filtering import count_by_period

        return count_by_period(self, period, start=start, end=end, rejected=rejected)

    def to_pandas(
        self,
        include_nested: bool = False,
        *,
        start: str | datetime | None = None,
        end: str | datetime | None = None,
        **kwargs: bool,
    ) -> PandasDataFrame:
        """Export dataset to pandas DataFrame.

        Args:
            include_nested: If True, include nested JSON columns (cvss_scores, etc.)
                           for full data fidelity. Default False excludes nested columns.
            start: Inclusive lower bound (>=) for nvd_published_at. ISO string or datetime.
            end: Exclusive upper bound (<) for nvd_published_at. ISO string or datetime.
            **kwargs: NVD status filters (analyzed, modified, awaiting, rejected,
                     deferred). Default excludes only rejected rows.

        Returns:
            Pandas DataFrame
        """
        from vulnflow.loaders.dataframe import PandasLoader

        return PandasLoader(
            include_nested=include_nested, start=start, end=end, **kwargs
        ).load(self)

    def to_polars(
        self,
        include_nested: bool = False,
        *,
        start: str | datetime | None = None,
        end: str | datetime | None = None,
        **kwargs: bool,
    ) -> PolarsDataFrame:
        """Export dataset to polars DataFrame.

        Args:
            include_nested: If True, include nested JSON columns (cvss_scores, etc.)
                           for full data fidelity. Default False excludes nested columns.
            start: Inclusive lower bound (>=) for nvd_published_at. ISO string or datetime.
            end: Exclusive upper bound (<) for nvd_published_at. ISO string or datetime.
            **kwargs: NVD status filters (analyzed, modified, awaiting, rejected,
                     deferred). Default excludes only rejected rows.

        Returns:
            Polars DataFrame
        """
        from vulnflow.loaders.dataframe import PolarsLoader

        return PolarsLoader(
            include_nested=include_nested, start=start, end=end, **kwargs
        ).load(self)

    def to_parquet(self, path: str | Path) -> None:
        """Export dataset to Parquet file (lossless, no filtering).

        Args:
            path: Output file path

        Raises:
            LoaderError: On any failure (permission denied, I/O error)
        """
        from vulnflow.exceptions import LoaderError
        from vulnflow.loaders.files import ParquetLoader

        try:
            loader = ParquetLoader(path)
            loader.load(self)
        except LoaderError as e:
            raise LoaderError(str(e)) from None
        except Exception as e:
            raise LoaderError(f"Failed to write Parquet to {path}: {e}") from None

    def to_json(self, path: str | Path) -> None:
        """Export dataset to JSON file (lossless, no filtering).

        Args:
            path: Output file path

        Raises:
            LoaderError: On any failure (permission denied, I/O error)
        """
        from vulnflow.exceptions import LoaderError
        from vulnflow.loaders.files import JSONLoader

        try:
            loader = JSONLoader(path)
            loader.load(self)
        except LoaderError as e:
            raise LoaderError(str(e)) from None
        except Exception as e:
            raise LoaderError(f"Failed to write JSON to {path}: {e}") from None

    @classmethod
    def from_msrc(
        cls,
        *,
        year: int | list[int] | None = None,
        start: str | None = None,
        end: str | None = None,
        cache_dir: str = ".cache/msrc",
        cache_ttl: int | None = None,
        timeout: float = 120.0,
    ) -> VulnDataset:
        """Load dataset from MSRC Patch Tuesday API.

        Args:
            year: Filter to specific year(s).
            start: Start month in YYYY-MM format (inclusive).
            end: End month in YYYY-MM format (inclusive). Requires start.
            cache_dir: Directory for HTTP response caching
            cache_ttl: HTTP cache TTL in seconds
            timeout: HTTP request timeout in seconds

        Returns:
            VulnDataset with MSRC vulnerability data

        Raises:
            ExtractorError: On HTTP failures or invalid parameters
        """
        from vulnflow.exceptions import ExtractorError
        from vulnflow.extractors.msrc import MSRCExtractor

        try:
            extractor = MSRCExtractor(
                year=year,
                start=start,
                end=end,
                cache_dir=cache_dir,
                cache_ttl=cache_ttl,
                timeout=timeout,
            )
            return extractor.extract()
        except ExtractorError as e:
            raise ExtractorError(str(e)) from None
        except Exception as e:
            raise ExtractorError(f"Failed to load MSRC data: {e}") from None

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
        from vulnflow.exceptions import ExtractorError
        from vulnflow.extractors.files import ParquetExtractor

        try:
            extractor = ParquetExtractor(path)
            return extractor.extract()
        except ExtractorError as e:
            raise ExtractorError(str(e)) from None
        except Exception as e:
            raise ExtractorError(f"Failed to load Parquet from {path}: {e}") from None

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
        from vulnflow.exceptions import ExtractorError
        from vulnflow.extractors.files import JSONExtractor

        try:
            extractor = JSONExtractor(str(path))
            return extractor.extract()
        except ExtractorError as e:
            raise ExtractorError(str(e)) from None
        except Exception as e:
            raise ExtractorError(f"Failed to load JSON from {path}: {e}") from None

    @classmethod
    def from_pandas(cls, dataframe: PandasDataFrame) -> VulnDataset:
        """Create dataset from pandas DataFrame (in-memory).

        Args:
            dataframe: Pandas DataFrame with flattened columns

        Returns:
            VulnDataset with data from DataFrame

        Raises:
            ExtractorError: On any failure (invalid input, conversion error)
        """
        from vulnflow.exceptions import ExtractorError
        from vulnflow.extractors.files import PandasExtractor

        try:
            extractor = PandasExtractor(dataframe)
            return extractor.extract()
        except ExtractorError as e:
            raise ExtractorError(str(e)) from None
        except Exception as e:
            raise ExtractorError(f"Failed to load from pandas DataFrame: {e}") from None

    def get_status_distribution(self) -> dict[str, int]:
        """Get distribution of nvd_status values.

        Returns:
            Dictionary mapping status values to counts

        Example:
            >>> ds = VulnDataset.from_nvd("data.json")
            >>> ds.get_status_distribution()
            {'analyzed': 1141, 'awaiting': 1506, 'modified': 89, 'rejected': 28}
        """
        from vulnflow.utils.filtering import status_distribution

        return status_distribution(self)

    @classmethod
    def from_polars(cls, dataframe: PolarsDataFrame) -> VulnDataset:
        """Create dataset from Polars DataFrame (in-memory).

        Args:
            dataframe: Polars DataFrame with flattened columns

        Returns:
            VulnDataset with data from DataFrame

        Raises:
            ExtractorError: On any failure (invalid input, conversion error)
        """
        from vulnflow.exceptions import ExtractorError
        from vulnflow.extractors.files import PolarsExtractor

        try:
            extractor = PolarsExtractor(dataframe)
            return extractor.extract()
        except ExtractorError as e:
            raise ExtractorError(str(e)) from None
        except Exception as e:
            raise ExtractorError(f"Failed to load from Polars DataFrame: {e}") from None
