"""
DataFrame Loaders - Pandas and Polars export

Provides:
- PandasLoader: Export to pandas DataFrame (with NVD status filtering)
- PolarsLoader: Export to polars DataFrame (with NVD status filtering)
- to_pandas: Legacy function for backward compatibility
- to_polars: Legacy function for backward compatibility

DataFrame exports are the *analysis path* — they filter to sane defaults.
File exports (parquet, json) are the *archive path* — always lossless.

Layer: Load
Dependencies: dataset
Used by: dataset.to_pandas(), to_polars()
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np

from vulnflow.registry import Registry

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

    from vulnflow.dataset import VulnDataset

# Columns containing nested JSON (can be excluded for cleaner DataFrames)
NESTED_COLUMNS = frozenset(
    {
        "cvss_scores",
        "cwes",
        "cpes",
        "references",
        "exploits",
        "epss_scores",
        "kev",
    }
)

# ============================================================================
# Status filtering (DataFrame loaders only)
# ============================================================================

_NVD_STATUS_DEFAULTS: dict[str, bool] = {
    "analyzed": True,
    "modified": True,
    "awaiting": True,
    "rejected": False,
    "deferred": True,
}


def _build_status_mask(
    dataset: VulnDataset,
    *,
    analyzed: bool = True,
    modified: bool = True,
    awaiting: bool = True,
    rejected: bool = False,
    deferred: bool = True,
) -> np.ndarray:
    """Build boolean mask based on NVD status values.

    Returns a boolean array where True means "include this row".
    If the dataset has no ``nvd_status`` column the mask is all-True (no-op).
    Rows with ``None`` status are always included.

    Args:
        dataset: VulnDataset to filter
        analyzed: Include rows with status 'analyzed'
        modified: Include rows with status 'modified'
        awaiting: Include rows with status 'awaiting'
        rejected: Include rows with status 'rejected'
        deferred: Include rows with status 'deferred'

    Returns:
        Boolean numpy array (dtype=bool), same length as dataset
    """
    statuses = dataset.get_column_or_none("nvd_status")
    if statuses is None:
        return np.ones(len(dataset), dtype=bool)

    flags = {
        "analyzed": analyzed,
        "modified": modified,
        "awaiting": awaiting,
        "rejected": rejected,
        "deferred": deferred,
    }
    included = [status for status, include in flags.items() if include]

    # None status → always include
    is_none = np.array([s is None for s in statuses], dtype=bool)
    mask: np.ndarray[tuple[int, ...], np.dtype[np.bool_]] = (
        np.isin(statuses, included) | is_none
    )
    return mask


# ============================================================================
# Date range filtering (DataFrame loaders only)
# ============================================================================


def _parse_date(value: str | datetime) -> datetime:
    """Parse a date string or pass through a datetime.

    Args:
        value: ISO date string (YYYY-MM-DD) or datetime

    Returns:
        Timezone-aware (UTC) datetime
    """
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def _build_date_mask(
    dataset: VulnDataset,
    *,
    start: str | datetime | None = None,
    end: str | datetime | None = None,
) -> np.ndarray:
    """Build boolean mask based on date range for nvd_published_at.

    Returns a boolean array where True means "include this row".
    If start/end are both None the mask is all-True (no-op).
    If the dataset has no ``nvd_published_at`` column the mask is all-True.

    Args:
        dataset: VulnDataset to filter
        start: Inclusive lower bound (>=). ISO string or datetime.
        end: Exclusive upper bound (<). ISO string or datetime.

    Returns:
        Boolean numpy array (dtype=bool), same length as dataset
    """
    n = len(dataset)
    if start is None and end is None:
        return np.ones(n, dtype=bool)

    dates = dataset.get_column_or_none("nvd_published_at")
    if dates is None:
        return np.ones(n, dtype=bool)

    # Handle datetime64 arrays (from Parquet round-trip)
    if np.issubdtype(dates.dtype, np.datetime64):
        mask = np.ones(n, dtype=bool)
        if start is not None:
            dt_start = _parse_date(start)
            np_start = np.datetime64(dt_start.replace(tzinfo=None), "ms")
            mask &= dates >= np_start
        if end is not None:
            dt_end = _parse_date(end)
            np_end = np.datetime64(dt_end.replace(tzinfo=None), "ms")
            mask &= dates < np_end
        return mask

    # Handle object arrays (Python datetime from NVD extractor)
    parsed_start: datetime | None = _parse_date(start) if start is not None else None
    parsed_end: datetime | None = _parse_date(end) if end is not None else None

    mask = np.empty(n, dtype=bool)
    for i in range(n):
        val = dates[i]
        if val is None:
            mask[i] = False
            continue
        # Make naive datetimes UTC-aware for comparison
        if isinstance(val, datetime) and val.tzinfo is None:
            val = val.replace(tzinfo=UTC)
        ok = True
        if parsed_start is not None:
            # Ensure both are aware or both naive for comparison
            cmp_start = parsed_start
            if val.tzinfo is not None and cmp_start.tzinfo is None:
                cmp_start = cmp_start.replace(tzinfo=UTC)
            ok = ok and val >= cmp_start
        if parsed_end is not None:
            cmp_end = parsed_end
            if val.tzinfo is not None and cmp_end.tzinfo is None:
                cmp_end = cmp_end.replace(tzinfo=UTC)
            ok = ok and val < cmp_end
        mask[i] = ok
    return mask


# ============================================================================
# DataFrame Loaders
# ============================================================================


@Registry.register_loader("pandas")
class PandasLoader:
    """Load VulnDataset to pandas DataFrame.

    By default, excludes nested JSON columns for cleaner DataFrames (lossy).
    Set include_nested=True for full data fidelity on round-trip.

    Filters rows by NVD status (analysis path). Default excludes only
    rejected rows.
    """

    format: str = "pandas"

    def __init__(
        self,
        include_nested: bool = False,
        *,
        start: str | datetime | None = None,
        end: str | datetime | None = None,
        **status_kwargs: bool,
    ) -> None:
        """Initialize loader.

        Args:
            include_nested: If True, include nested JSON columns for full fidelity.
                           Default False excludes nested columns (lossy by default).
            start: Inclusive lower bound (>=) for nvd_published_at. ISO string or datetime.
            end: Exclusive upper bound (<) for nvd_published_at. ISO string or datetime.
            **status_kwargs: Override NVD status filters (analyzed, modified,
                            awaiting, rejected, deferred). Default excludes
                            only rejected rows.
        """
        self.include_nested = include_nested
        self._start = start
        self._end = end
        self._status_kwargs = {
            k: status_kwargs.get(k, v) for k, v in _NVD_STATUS_DEFAULTS.items()
        }

    def load(self, dataset: VulnDataset) -> pd.DataFrame:
        """Load dataset to pandas DataFrame.

        Args:
            dataset: VulnDataset to export

        Returns:
            Pandas DataFrame
        """
        import pandas as pd

        mask = _build_status_mask(dataset, **self._status_kwargs) & _build_date_mask(
            dataset, start=self._start, end=self._end
        )
        if not self.include_nested:
            data = {
                k: v[mask] for k, v in dataset._data.items() if k not in NESTED_COLUMNS
            }
            return pd.DataFrame(data)

        return pd.DataFrame({k: v[mask] for k, v in dataset._data.items()})


@Registry.register_loader("polars")
class PolarsLoader:
    """Load VulnDataset to polars DataFrame.

    Builds directly from numpy arrays for performance (no pandas intermediate).

    By default, excludes nested JSON columns for cleaner DataFrames (lossy).
    Set include_nested=True for full data fidelity on round-trip.

    Filters rows by NVD status (analysis path). Default excludes only
    rejected rows.
    """

    format: str = "polars"

    def __init__(
        self,
        include_nested: bool = False,
        *,
        start: str | datetime | None = None,
        end: str | datetime | None = None,
        **status_kwargs: bool,
    ) -> None:
        """Initialize loader.

        Args:
            include_nested: If True, include nested JSON columns for full fidelity.
                           Default False excludes nested columns (lossy by default).
            start: Inclusive lower bound (>=) for nvd_published_at. ISO string or datetime.
            end: Exclusive upper bound (<) for nvd_published_at. ISO string or datetime.
            **status_kwargs: Override NVD status filters (analyzed, modified,
                            awaiting, rejected, deferred). Default excludes
                            only rejected rows.
        """
        self.include_nested = include_nested
        self._start = start
        self._end = end
        self._status_kwargs = {
            k: status_kwargs.get(k, v) for k, v in _NVD_STATUS_DEFAULTS.items()
        }

    def load(self, dataset: VulnDataset) -> pl.DataFrame:
        """Load dataset to polars DataFrame.

        Args:
            dataset: VulnDataset to export

        Returns:
            Polars DataFrame
        """
        import polars as pl

        mask = _build_status_mask(dataset, **self._status_kwargs) & _build_date_mask(
            dataset, start=self._start, end=self._end
        )
        if not self.include_nested:
            data = {
                k: v[mask].tolist()
                for k, v in dataset._data.items()
                if k not in NESTED_COLUMNS
            }
        else:
            data = {k: v[mask].tolist() for k, v in dataset._data.items()}

        return pl.from_dict(data)


# ============================================================================
# Legacy Functions (backward compatibility)
# ============================================================================


def to_pandas(dataset: VulnDataset) -> pd.DataFrame:
    """Export dataset to pandas DataFrame.

    LEGACY: Use PandasLoader class for new code.

    Args:
        dataset: VulnDataset to export

    Returns:
        Pandas DataFrame with all columns (no flattening)
    """
    import pandas as pd

    return pd.DataFrame(dataset._data)


def to_polars(dataset: VulnDataset) -> pl.DataFrame:
    """Export dataset to polars DataFrame.

    LEGACY: Use PolarsLoader class for new code.

    Args:
        dataset: VulnDataset to export

    Returns:
        Polars DataFrame with all columns (no flattening)
    """
    import polars as pl

    # Convert numpy arrays to lists for polars compatibility
    data = {k: v.tolist() for k, v in dataset._data.items()}
    return pl.from_dict(data)
