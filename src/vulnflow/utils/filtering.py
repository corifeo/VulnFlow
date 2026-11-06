"""
Filtering utilities - status masks, date masks, and period grouping

Provides:
- _NVD_STATUS_DEFAULTS: Default inclusion flags for NVD statuses
- _build_status_mask: Boolean mask based on NVD status values
- _parse_date: Parse date strings or datetimes to UTC
- _build_date_mask: Boolean mask based on date range
- _VALID_PERIODS: Valid period names for count_by
- _extract_period_keys: Convert datetime array to period string keys
- count_by_period: Count records grouped by time period
- status_distribution: Get distribution of nvd_status values

Layer: Utility
Dependencies: numpy
Used by: dataset, loaders/dataframe
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

    from vulnflow.dataset import VulnDataset

# ============================================================================
# Status filtering
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
# Date range filtering
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
# Period grouping
# ============================================================================

_VALID_PERIODS = frozenset({"year", "quarter", "month", "week", "day"})


def _extract_period_keys(dates: np.ndarray, period: str) -> np.ndarray:
    """Convert datetime array to string period keys.

    Handles both Python datetime objects and numpy datetime64 values.

    Args:
        dates: Array of datetime objects (after mask applied, no Nones)
        period: One of 'year', 'quarter', 'month', 'week', 'day'

    Returns:
        Object array of string keys (e.g. '2024-01', '2024-Q1')
    """
    n = len(dates)
    keys = np.empty(n, dtype=object)

    for i in range(n):
        val = dates[i]
        # Convert datetime64 to Python datetime if needed
        if isinstance(val, np.datetime64):
            # Convert to timestamp then to datetime
            ts = (val - np.datetime64("1970-01-01T00:00:00", "ms")) / np.timedelta64(
                1, "ms"
            )
            dt = datetime.fromtimestamp(float(ts) / 1000.0, tz=UTC)
        else:
            dt = val

        if period == "year":
            keys[i] = f"{dt.year}"
        elif period == "quarter":
            q = (dt.month - 1) // 3 + 1
            keys[i] = f"{dt.year}-Q{q}"
        elif period == "month":
            keys[i] = f"{dt.year}-{dt.month:02d}"
        elif period == "week":
            iso_year, iso_week, _ = dt.isocalendar()
            keys[i] = f"{iso_year}-W{iso_week:02d}"
        else:  # day
            keys[i] = f"{dt.year}-{dt.month:02d}-{dt.day:02d}"

    return keys


# ============================================================================
# Aggregate helpers (used by VulnDataset thin wrappers)
# ============================================================================


def count_by_period(
    dataset: VulnDataset,
    period: str = "month",
    *,
    start: str | datetime | None = None,
    end: str | datetime | None = None,
    rejected: bool = False,
) -> pd.DataFrame:
    """Count records grouped by time period.

    Args:
        dataset: VulnDataset to count
        period: Grouping period — 'year', 'quarter', 'month', 'week', or 'day'.
        start: Inclusive lower bound (>=) for nvd_published_at.
        end: Exclusive upper bound (<) for nvd_published_at.
        rejected: Include rejected rows (default False).

    Returns:
        Pandas DataFrame with columns [period, 'count'], sorted chronologically.

    Raises:
        ValueError: If period is not a valid option.
    """
    import pandas as pd

    if period not in _VALID_PERIODS:
        msg = f"Invalid period {period!r}, must be one of {sorted(_VALID_PERIODS)}"
        raise ValueError(msg)

    # Build combined mask: status + date range
    status_mask = _build_status_mask(dataset, rejected=rejected)
    date_mask = _build_date_mask(dataset, start=start, end=end)
    mask = status_mask & date_mask

    dates = dataset.get_column_or_none("nvd_published_at")
    if dates is None:
        return pd.DataFrame(
            {period: pd.Series(dtype=object), "count": pd.Series(dtype=int)}
        )

    filtered_dates = dates[mask]
    if len(filtered_dates) == 0:
        return pd.DataFrame(
            {period: pd.Series(dtype=object), "count": pd.Series(dtype=int)}
        )

    keys = _extract_period_keys(filtered_dates, period)
    unique_keys, counts = np.unique(keys, return_counts=True)

    # np.unique sorts lexicographically, which is chronological for our formats
    return pd.DataFrame({period: unique_keys, "count": counts})


def status_distribution(dataset: VulnDataset) -> dict[str, int]:
    """Get distribution of nvd_status values.

    Args:
        dataset: VulnDataset to analyze

    Returns:
        Dictionary mapping status values to counts
    """
    statuses = dataset.get_column("nvd_status")
    unique, counts = np.unique(statuses, return_counts=True)
    return dict(zip(unique.tolist(), counts.tolist(), strict=True))
