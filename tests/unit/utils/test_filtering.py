"""
Filtering utilities - standalone function tests

Tests: count_by_period, status_distribution, _extract_period_keys

Layer: Utility
Dependencies: dataset
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from vulnflow.dataset import VulnDataset
from vulnflow.utils.filtering import (
    _VALID_PERIODS,
    _extract_period_keys,
    count_by_period,
    status_distribution,
)

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
    return datetime(year, month, day, tzinfo=UTC)


# ============================================================================
# _extract_period_keys
# ============================================================================


class TestExtractPeriodKeys:
    """Tests for _extract_period_keys standalone function."""

    def test_month_keys(self) -> None:
        dates = np.array([_utc(2024, 1, 15), _utc(2024, 3, 20)], dtype=object)
        keys = _extract_period_keys(dates, "month")
        assert keys.tolist() == ["2024-01", "2024-03"]

    def test_year_keys(self) -> None:
        dates = np.array([_utc(2023, 6, 1), _utc(2024, 1, 1)], dtype=object)
        keys = _extract_period_keys(dates, "year")
        assert keys.tolist() == ["2023", "2024"]

    def test_quarter_keys(self) -> None:
        dates = np.array([_utc(2024, 2, 1), _utc(2024, 7, 1)], dtype=object)
        keys = _extract_period_keys(dates, "quarter")
        assert keys.tolist() == ["2024-Q1", "2024-Q3"]

    def test_week_keys(self) -> None:
        dates = np.array([_utc(2024, 1, 1)], dtype=object)
        keys = _extract_period_keys(dates, "week")
        assert keys[0].startswith("2024-W")

    def test_day_keys(self) -> None:
        dates = np.array([_utc(2024, 3, 5)], dtype=object)
        keys = _extract_period_keys(dates, "day")
        assert keys.tolist() == ["2024-03-05"]

    def test_datetime64_input(self) -> None:
        dates = np.array(["2024-06-15"], dtype="datetime64[ms]")
        keys = _extract_period_keys(dates, "month")
        assert keys.tolist() == ["2024-06"]


# ============================================================================
# count_by_period
# ============================================================================


class TestCountByPeriod:
    """Tests for count_by_period standalone function."""

    def test_basic_monthly_count(self) -> None:
        ds = _dataset_with_dates([
            _utc(2024, 1, 1),
            _utc(2024, 1, 15),
            _utc(2024, 3, 1),
        ])
        df = count_by_period(ds, "month")
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["month", "count"]
        assert len(df) == 2
        assert df[df["month"] == "2024-01"]["count"].iloc[0] == 2

    def test_invalid_period_raises(self) -> None:
        ds = _dataset_with_dates([_utc(2024, 1, 1)])
        try:
            count_by_period(ds, "century")
            msg = "Expected ValueError"
            raise AssertionError(msg)
        except ValueError:
            pass

    def test_valid_periods_constant(self) -> None:
        assert _VALID_PERIODS == {"year", "quarter", "month", "week", "day"}

    def test_rejected_excluded_by_default(self) -> None:
        ds = _dataset_with_dates(
            [_utc(2024, 1, 1), _utc(2024, 1, 2)],
            statuses=["analyzed", "rejected"],
        )
        df = count_by_period(ds, "month")
        assert df["count"].iloc[0] == 1

    def test_rejected_included_when_requested(self) -> None:
        ds = _dataset_with_dates(
            [_utc(2024, 1, 1), _utc(2024, 1, 2)],
            statuses=["analyzed", "rejected"],
        )
        df = count_by_period(ds, "month", rejected=True)
        assert df["count"].iloc[0] == 2

    def test_date_range_filter(self) -> None:
        ds = _dataset_with_dates([
            _utc(2024, 1, 1),
            _utc(2024, 6, 1),
            _utc(2024, 12, 1),
        ])
        df = count_by_period(ds, "month", start="2024-06-01", end="2024-12-01")
        assert len(df) == 1
        assert df["month"].iloc[0] == "2024-06"

    def test_no_dates_column(self) -> None:
        ds = VulnDataset()
        ds.set_column("cve_id", np.array(["CVE-1"], dtype=object))
        df = count_by_period(ds, "month")
        assert len(df) == 0

    def test_empty_after_filter(self) -> None:
        ds = _dataset_with_dates(
            [_utc(2024, 1, 1)],
            statuses=["rejected"],
        )
        df = count_by_period(ds, "month")
        assert len(df) == 0


# ============================================================================
# status_distribution
# ============================================================================


class TestStatusDistribution:
    """Tests for status_distribution standalone function."""

    def test_basic_distribution(self) -> None:
        ds = VulnDataset()
        ds.set_column("cve_id", np.array(["CVE-1", "CVE-2", "CVE-3"], dtype=object))
        ds.set_column(
            "nvd_status",
            np.array(["analyzed", "analyzed", "rejected"], dtype=object),
        )
        result = status_distribution(ds)
        assert result == {"analyzed": 2, "rejected": 1}

    def test_single_status(self) -> None:
        ds = VulnDataset()
        ds.set_column("cve_id", np.array(["CVE-1"], dtype=object))
        ds.set_column("nvd_status", np.array(["modified"], dtype=object))
        result = status_distribution(ds)
        assert result == {"modified": 1}
