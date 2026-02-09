"""
Date Range Filters - start/end filtering for DataFrame loaders

Tests: _build_date_mask, PandasLoader date range, PolarsLoader date range

Layer: Load
Dependencies: dataset, loaders
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import polars as pl

from vulnflow.dataset import VulnDataset
from vulnflow.loaders.dataframe import _build_date_mask

# ============================================================================
# Fixture helpers
# ============================================================================


def _dataset_with_dates(
    dates: list[datetime | None],
    statuses: list[str] | None = None,
) -> VulnDataset:
    """Build a minimal VulnDataset with given nvd_published_at values."""
    ds = VulnDataset()
    n = len(dates)
    ds.set_column(
        "cve_id", np.array([f"CVE-2024-{i:04d}" for i in range(n)], dtype=object)
    )
    ds.set_column("nvd_published_at", np.array(dates, dtype=object))
    if statuses is not None:
        ds.set_column("nvd_status", np.array(statuses, dtype=object))
    return ds


def _utc(year: int, month: int, day: int) -> datetime:
    """Shorthand for UTC-aware datetime."""
    return datetime(year, month, day, tzinfo=UTC)


# ============================================================================
# _build_date_mask unit tests
# ============================================================================


class TestBuildDateMask:
    """Tests for _build_date_mask() boolean mask builder."""

    def test_no_range_returns_all_true(self) -> None:
        """No start/end → all-True mask."""
        ds = _dataset_with_dates([_utc(2024, 1, 1), _utc(2024, 6, 1)])
        mask = _build_date_mask(ds)
        assert mask.dtype == bool
        assert mask.all()

    def test_start_only(self) -> None:
        """start= filters out dates before start (inclusive >=)."""
        ds = _dataset_with_dates([
            _utc(2024, 1, 1),
            _utc(2024, 6, 1),
            _utc(2024, 12, 1),
        ])
        mask = _build_date_mask(ds, start="2024-06-01")
        assert mask.tolist() == [False, True, True]

    def test_end_only(self) -> None:
        """end= filters out dates at or after end (exclusive <)."""
        ds = _dataset_with_dates([
            _utc(2024, 1, 1),
            _utc(2024, 6, 1),
            _utc(2024, 12, 1),
        ])
        mask = _build_date_mask(ds, end="2024-06-01")
        assert mask.tolist() == [True, False, False]

    def test_start_and_end(self) -> None:
        """Both start and end create a half-open interval [start, end)."""
        ds = _dataset_with_dates([
            _utc(2024, 1, 1),
            _utc(2024, 6, 1),
            _utc(2024, 12, 1),
        ])
        mask = _build_date_mask(ds, start="2024-06-01", end="2024-12-01")
        assert mask.tolist() == [False, True, False]

    def test_string_parsing(self) -> None:
        """String dates are parsed correctly."""
        ds = _dataset_with_dates([_utc(2024, 3, 15)])
        mask = _build_date_mask(ds, start="2024-03-15")
        assert mask.tolist() == [True]

    def test_datetime_input(self) -> None:
        """datetime objects accepted as start/end."""
        ds = _dataset_with_dates([_utc(2024, 3, 15)])
        mask = _build_date_mask(ds, start=_utc(2024, 3, 15))
        assert mask.tolist() == [True]

    def test_none_dates_excluded_when_range_specified(self) -> None:
        """None dates are excluded when a date range is specified."""
        ds = _dataset_with_dates([_utc(2024, 6, 1), None])
        mask = _build_date_mask(ds, start="2024-01-01")
        assert mask.tolist() == [True, False]

    def test_no_column_returns_all_true(self) -> None:
        """Dataset without nvd_published_at → all-True mask."""
        ds = VulnDataset()
        ds.set_column("cve_id", np.array(["CVE-1", "CVE-2"], dtype=object))
        mask = _build_date_mask(ds, start="2024-01-01")
        assert mask.all()

    def test_datetime64_support(self) -> None:
        """datetime64 arrays (from Parquet round-trip) work correctly."""
        ds = VulnDataset()
        ds.set_column("cve_id", np.array(["CVE-1", "CVE-2"], dtype=object))
        dates = np.array(["2024-01-15", "2024-06-15"], dtype="datetime64[ms]")
        ds.set_column("nvd_published_at", dates)
        mask = _build_date_mask(ds, start="2024-06-01")
        assert mask.tolist() == [False, True]

    def test_none_dates_no_range_included(self) -> None:
        """None dates without range specified are included (all-True)."""
        ds = _dataset_with_dates([_utc(2024, 6, 1), None])
        mask = _build_date_mask(ds)
        assert mask.all()


# ============================================================================
# PandasLoader date range integration
# ============================================================================


class TestPandasLoaderDateRange:
    """Tests for PandasLoader start/end date filtering."""

    def test_start_filter(self) -> None:
        """to_pandas(start=...) filters by date."""
        ds = _dataset_with_dates([
            _utc(2024, 1, 1),
            _utc(2024, 6, 1),
            _utc(2024, 12, 1),
        ])
        df = ds.to_pandas(start="2024-06-01")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_end_filter(self) -> None:
        """to_pandas(end=...) filters by date."""
        ds = _dataset_with_dates([
            _utc(2024, 1, 1),
            _utc(2024, 6, 1),
        ])
        df = ds.to_pandas(end="2024-06-01")
        assert len(df) == 1

    def test_combined_date_and_status(self) -> None:
        """Date filter AND status filter compose together."""
        ds = _dataset_with_dates(
            [_utc(2024, 1, 1), _utc(2024, 6, 1), _utc(2024, 12, 1)],
            statuses=["analyzed", "rejected", "analyzed"],
        )
        # Start=June, rejected=False → only Dec row survives
        df = ds.to_pandas(start="2024-06-01")
        assert len(df) == 1

    def test_no_params_unchanged_behavior(self) -> None:
        """to_pandas() without date params behaves as before."""
        ds = _dataset_with_dates([_utc(2024, 1, 1), _utc(2024, 6, 1)])
        df = ds.to_pandas()
        assert len(df) == 2


# ============================================================================
# PolarsLoader date range integration
# ============================================================================


class TestPolarsLoaderDateRange:
    """Tests for PolarsLoader start/end date filtering."""

    def test_start_filter(self) -> None:
        """to_polars(start=...) filters by date."""
        ds = _dataset_with_dates([
            _utc(2024, 1, 1),
            _utc(2024, 6, 1),
            _utc(2024, 12, 1),
        ])
        df = ds.to_polars(start="2024-06-01")
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 2

    def test_start_and_end_filter(self) -> None:
        """to_polars(start=..., end=...) filters by date range."""
        ds = _dataset_with_dates([
            _utc(2024, 1, 1),
            _utc(2024, 6, 1),
            _utc(2024, 12, 1),
        ])
        df = ds.to_polars(start="2024-06-01", end="2024-12-01")
        assert len(df) == 1
