"""
Merge utilities - row selection for deduplicating merges

Tests: latest_row_indices

Layer: Utility
Dependencies: numpy
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from vulnflow.utils.merge import latest_row_indices


def _ids(*values: str) -> np.ndarray:
    return np.array(values, dtype=object)


def _dt(day: int) -> datetime:
    return datetime(2026, 11, day, tzinfo=UTC)


class TestLatestRowIndices:
    def test_unique_ids_returns_all_rows_in_order(self) -> None:
        idx = latest_row_indices(_ids("CVE-1", "CVE-2", "CVE-3"), None)
        assert idx.tolist() == [0, 1, 2]

    def test_duplicate_without_dates_later_row_wins(self) -> None:
        idx = latest_row_indices(_ids("CVE-1", "CVE-2", "CVE-1"), None)
        assert idx.tolist() == [2, 1]

    def test_newest_modified_wins_even_when_earlier(self) -> None:
        modified = np.array([_dt(9), _dt(5)], dtype=object)
        idx = latest_row_indices(_ids("CVE-1", "CVE-1"), modified)
        assert idx.tolist() == [0]

    def test_equal_modified_later_row_wins(self) -> None:
        modified = np.array([_dt(5), _dt(5)], dtype=object)
        idx = latest_row_indices(_ids("CVE-1", "CVE-1"), modified)
        assert idx.tolist() == [1]

    def test_missing_date_falls_back_to_later_row(self) -> None:
        """Dates only decide when every copy has one; otherwise the later row wins."""
        modified = np.array([_dt(5), None], dtype=object)
        assert latest_row_indices(_ids("CVE-1", "CVE-1"), modified).tolist() == [1]
        modified = np.array([None, _dt(5)], dtype=object)
        assert latest_row_indices(_ids("CVE-1", "CVE-1"), modified).tolist() == [1]

    def test_missing_date_on_one_cve_does_not_affect_others(self) -> None:
        modified = np.array([_dt(9), _dt(5), _dt(1), None], dtype=object)
        idx = latest_row_indices(_ids("CVE-1", "CVE-1", "CVE-2", "CVE-2"), modified)
        assert idx.tolist() == [0, 3]

    def test_datetime64_column_nat_counts_as_missing(self) -> None:
        modified = np.array(["2026-11-09", "NaT", "2026-11-01"], dtype="datetime64[us]")
        idx = latest_row_indices(_ids("CVE-1", "CVE-1", "CVE-2"), modified)
        assert idx.tolist() == [1, 2]

    def test_naive_and_aware_datetimes_compare_as_utc(self) -> None:
        modified = np.array([datetime(2026, 11, 9), _dt(5)], dtype=object)
        idx = latest_row_indices(_ids("CVE-1", "CVE-1"), modified)
        assert idx.tolist() == [0]

    def test_result_follows_first_appearance_of_each_id(self) -> None:
        idx = latest_row_indices(_ids("CVE-2", "CVE-1", "CVE-2", "CVE-3"), None)
        assert idx.tolist() == [2, 1, 3]

    def test_empty_input(self) -> None:
        assert latest_row_indices(_ids(), None).tolist() == []
