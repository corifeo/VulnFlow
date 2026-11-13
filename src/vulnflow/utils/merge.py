"""
Merge utilities - pick one row per CVE when datasets overlap

Provides:
- latest_row_indices: Row indices to keep, one per cve_id (newest modified, else later)
- dedupe_columns: Apply latest_row_indices to a column dict

Layer: Utility
Dependencies: numpy, utils.nvd
Used by: dataset, extractors/files
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vulnflow.utils.nvd import to_utc_datetime

MODIFIED_COLUMN = "nvd_modified_at"
_OLDEST = np.iinfo(np.int64).min  # missing dates sort first, so any date beats them


def _modified_keys(
    modified: np.ndarray[Any, Any] | None, n: int
) -> np.ndarray[Any, Any]:
    """Comparable int64 microseconds since epoch (UTC); missing values become _OLDEST."""
    if modified is None:
        return np.full(n, _OLDEST, dtype=np.int64)
    if np.issubdtype(modified.dtype, np.datetime64):
        # NaT is int64 min, i.e. already _OLDEST
        return modified.astype("datetime64[us]").view(np.int64)
    keys = np.full(n, _OLDEST, dtype=np.int64)
    for i, value in enumerate(modified):
        ts = to_utc_datetime(value)
        if ts is not None:
            keys[i] = int(ts.timestamp() * 1_000_000)
    return keys


def latest_row_indices(
    cve_ids: np.ndarray[Any, Any], modified: np.ndarray[Any, Any] | None
) -> np.ndarray[Any, Any]:
    """Indices of the rows to keep: one per cve_id.

    The row with the newest ``modified`` timestamp wins; ties go to the
    later row. Timestamps only decide when every row of that CVE has one:
    if any is missing, the later row wins. Naive datetimes are read as UTC.

    Args:
        cve_ids: CVE ID per row.
        modified: Last-modified timestamp per row (datetime64, datetimes,
            ISO strings or None), or None to keep the last occurrence.

    Returns:
        Kept row indices, ordered by each CVE's first appearance.
    """
    n = len(cve_ids)
    if n == 0:
        return np.array([], dtype=np.intp)
    ids = np.asarray(cve_ids).astype(str)
    order = np.arange(n)
    keys = _modified_keys(modified, n)
    missing = keys == _OLDEST
    if missing.any():  # a CVE with any undated row falls back to position
        keys = np.where(np.isin(ids, ids[missing]), _OLDEST, keys)
    ranked = np.lexsort((order, keys, ids))  # by id, then date, then position
    sorted_ids = ids[ranked]
    is_last = np.append(sorted_ids[1:] != sorted_ids[:-1], True)
    is_first = np.insert(sorted_ids[1:] != sorted_ids[:-1], 0, True)
    winners = ranked[is_last]
    first_seen = np.minimum.reduceat(ranked, np.flatnonzero(is_first))
    kept: np.ndarray[Any, Any] = winners[np.argsort(first_seen, kind="stable")]
    return kept


def dedupe_columns(
    columns: dict[str, np.ndarray[Any, Any]],
) -> dict[str, np.ndarray[Any, Any]]:
    """Keep one row per cve_id (see latest_row_indices); unchanged without a cve_id column."""
    if "cve_id" not in columns:
        return columns
    keep = latest_row_indices(columns["cve_id"], columns.get(MODIFIED_COLUMN))
    if len(keep) == len(columns["cve_id"]):
        return columns
    return {name: values[keep] for name, values in columns.items()}
