"""
Tests for vulnflow.helpers.explore module.

Tests:
- load() cache resolution and file extension dispatch
- status_distribution() delegation
- severity_distribution() value counts
- top_values() frequency counting
- yearly_counts() year extraction from cve_id
- monthly_counts() with optional year filter
- inspect_nested() JSON column parsing
- find() multi-criteria filtering
- summary() structured overview
- describe_severity() severity dict with per-year detail
- describe_yearly() year counts as dict
- describe_monthly() month counts as dict
- describe_nested() nested field counts as dict
- search() filter and return dicts
- lookup() single CVE as dict
- data_info() cache status reporting
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from vulnflow.dataset import VulnDataset
from vulnflow.helpers.explore import (
    _extract_field,
    _parse_json_cell,
    data_info,
    describe_monthly,
    describe_nested,
    describe_severity,
    describe_yearly,
    find,
    inspect_nested,
    load,
    lookup,
    monthly_counts,
    search,
    severity_distribution,
    status_distribution,
    summary,
    top_values,
    yearly_counts,
)

# ============================================================================
# Fixtures
# ============================================================================


def _make_dataset(columns: dict[str, list[Any]]) -> VulnDataset:
    """Create a VulnDataset from column dicts for testing."""
    ds = VulnDataset()
    ds._data = {k: np.array(v, dtype=object) for k, v in columns.items()}
    return ds


def _make_df(columns: dict[str, list[Any]]) -> pd.DataFrame:
    """Create a DataFrame from column dicts for testing."""
    return pd.DataFrame(columns)


@pytest.fixture
def sample_dataset() -> VulnDataset:
    """Dataset with basic columns for testing."""
    return _make_dataset(
        {
            "cve_id": [
                "CVE-2023-0001",
                "CVE-2023-0002",
                "CVE-2024-0001",
                "CVE-2024-0002",
                "CVE-2024-0003",
            ],
            "nvd_status": [
                "analyzed",
                "analyzed",
                "modified",
                "analyzed",
                "rejected",
            ],
            "description": [
                "SQL injection in login",
                "XSS in search page",
                "Buffer overflow in parser",
                "Remote code execution",
                "Not a vulnerability",
            ],
            "nvd_published_at": [
                datetime(2023, 3, 15),
                datetime(2023, 7, 20),
                datetime(2024, 1, 10),
                datetime(2024, 6, 5),
                datetime(2024, 6, 15),
            ],
            "cvss_base_severity": [
                "HIGH",
                "MEDIUM",
                "CRITICAL",
                "HIGH",
                None,
            ],
            "cvss_base_score": [7.5, 5.3, 9.8, 8.1, None],
            "primary_cwe_id": [
                "CWE-89",
                "CWE-79",
                "CWE-120",
                "CWE-89",
                None,
            ],
        }
    )


@pytest.fixture
def sample_df(sample_dataset: VulnDataset) -> pd.DataFrame:
    """DataFrame version of sample_dataset."""
    return pd.DataFrame(sample_dataset._data)


@pytest.fixture
def nested_dataset() -> VulnDataset:
    """Dataset with JSON-encoded nested columns."""
    cwes = [
        json.dumps([{"id": "CWE-89", "name": "SQL Injection"}]),
        json.dumps([{"id": "CWE-79", "name": "XSS"}]),
        json.dumps(
            [
                {"id": "CWE-89", "name": "SQL Injection"},
                {"id": "CWE-120", "name": "Buffer Overflow"},
            ]
        ),
    ]
    cvss = [
        json.dumps([{"source": "nvd", "base_score": 7.5}]),
        json.dumps(
            [
                {"source": "nvd", "base_score": 5.3},
                {"source": "redhat", "base_score": 5.0},
            ]
        ),
        json.dumps([{"source": "redhat", "base_score": 9.8}]),
    ]
    return _make_dataset(
        {
            "cve_id": ["CVE-2024-0001", "CVE-2024-0002", "CVE-2024-0003"],
            "cwes": cwes,
            "cvss_scores": cvss,
        }
    )


# ============================================================================
# Tests: load()
# ============================================================================


class TestLoad:
    def test_load_explicit_parquet(self, tmp_path: Path) -> None:
        """load() with explicit .parquet path loads from parquet."""
        ds = _make_dataset({"cve_id": ["CVE-2024-0001"]})
        parquet_path = tmp_path / "test.parquet"
        ds.to_parquet(parquet_path)

        result = load(str(parquet_path))
        assert len(result) == 1

    def test_load_explicit_path_not_found_raises(self) -> None:
        """load() with non-existent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load("nonexistent.parquet")

    def test_load_unsupported_extension_raises(self, tmp_path: Path) -> None:
        """load() with unsupported extension raises ValueError."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("a,b\n1,2")
        with pytest.raises(ValueError, match="Unsupported file extension"):
            load(str(csv_path))

    def test_load_explicit_json_roundtrip(self, tmp_path: Path) -> None:
        """load() with .json path loads JSON file."""
        ds = _make_dataset({"cve_id": ["CVE-2024-0001"]})
        json_path = tmp_path / "test.json"
        ds.to_json(json_path)

        result = load(str(json_path))
        assert len(result) == 1


# ============================================================================
# Tests: status_distribution()
# ============================================================================


class TestStatusDistribution:
    def test_returns_correct_counts(self, sample_dataset: VulnDataset) -> None:
        """status_distribution returns correct counts per status."""
        dist = status_distribution(sample_dataset)
        assert dist["analyzed"] == 3
        assert dist["modified"] == 1
        assert dist["rejected"] == 1


# ============================================================================
# Tests: severity_distribution()
# ============================================================================


class TestSeverityDistribution:
    def test_returns_value_counts(self, sample_df: pd.DataFrame) -> None:
        """severity_distribution returns counts per severity level."""
        result = severity_distribution(sample_df)
        assert result["HIGH"] == 2
        assert result["MEDIUM"] == 1
        assert result["CRITICAL"] == 1

    def test_missing_column_raises(self) -> None:
        """severity_distribution raises KeyError if column missing."""
        df = pd.DataFrame({"cve_id": ["CVE-2024-0001"]})
        with pytest.raises(KeyError, match="cvss_base_severity"):
            severity_distribution(df)


# ============================================================================
# Tests: top_values()
# ============================================================================


class TestTopValues:
    def test_returns_top_n(self, sample_df: pd.DataFrame) -> None:
        """top_values returns the most frequent values."""
        result = top_values(sample_df, "cvss_base_severity", n=2)
        assert len(result) == 2
        assert result.index[0] == "HIGH"

    def test_missing_column_raises(self, sample_df: pd.DataFrame) -> None:
        """top_values raises KeyError for missing column."""
        with pytest.raises(KeyError, match="nonexistent"):
            top_values(sample_df, "nonexistent")


# ============================================================================
# Tests: yearly_counts()
# ============================================================================


class TestYearlyCounts:
    def test_extracts_year_from_cve_id(self, sample_df: pd.DataFrame) -> None:
        """yearly_counts extracts years correctly from CVE IDs."""
        result = yearly_counts(sample_df)
        assert result[2023] == 2
        assert result[2024] == 3

    def test_sorted_by_year(self, sample_df: pd.DataFrame) -> None:
        """yearly_counts returns results sorted by year."""
        result = yearly_counts(sample_df)
        years = list(result.index)
        assert years == sorted(years)

    def test_missing_column_raises(self) -> None:
        """yearly_counts raises KeyError if cve_id missing."""
        df = pd.DataFrame({"description": ["test"]})
        with pytest.raises(KeyError, match="cve_id"):
            yearly_counts(df)


# ============================================================================
# Tests: monthly_counts()
# ============================================================================


class TestMonthlyCounts:
    def test_counts_all_months(self, sample_df: pd.DataFrame) -> None:
        """monthly_counts returns counts for all months."""
        result = monthly_counts(sample_df)
        assert len(result) > 0

    def test_year_filter(self, sample_df: pd.DataFrame) -> None:
        """monthly_counts filters by year when specified."""
        result = monthly_counts(sample_df, year=2024)
        # 2024 has 3 entries: Jan, Jun, Jun
        assert result.sum() == 3

    def test_missing_column_raises(self) -> None:
        """monthly_counts raises KeyError if nvd_published_at missing."""
        df = pd.DataFrame({"cve_id": ["CVE-2024-0001"]})
        with pytest.raises(KeyError, match="nvd_published_at"):
            monthly_counts(df)


# ============================================================================
# Tests: inspect_nested()
# ============================================================================


class TestInspectNested:
    def test_extract_field_from_json_list(
        self, nested_dataset: VulnDataset
    ) -> None:
        """inspect_nested extracts field values from JSON arrays."""
        result = inspect_nested(nested_dataset, "cwes", "id")
        assert result["CWE-89"] == 2
        assert result["CWE-79"] == 1
        assert result["CWE-120"] == 1

    def test_extract_nested_source(self, nested_dataset: VulnDataset) -> None:
        """inspect_nested works for CVSS source extraction."""
        result = inspect_nested(nested_dataset, "cvss_scores", "source")
        assert result["nvd"] == 2
        assert result["redhat"] == 2

    def test_top_n_limits_results(self, nested_dataset: VulnDataset) -> None:
        """inspect_nested respects the n parameter."""
        result = inspect_nested(nested_dataset, "cwes", "id", n=2)
        assert len(result) == 2

    def test_no_field_returns_stringified_objects(
        self, nested_dataset: VulnDataset
    ) -> None:
        """inspect_nested without field returns stringified entries."""
        result = inspect_nested(nested_dataset, "cwes")
        assert len(result) > 0

    def test_empty_column_returns_empty_series(self) -> None:
        """inspect_nested returns empty Series for all-null column."""
        ds = _make_dataset({"cwes": [None, "null", "[]"]})
        result = inspect_nested(ds, "cwes", "id")
        assert len(result) == 0

    def test_missing_column_raises(self) -> None:
        """inspect_nested raises KeyError for missing column."""
        ds = _make_dataset({"cve_id": ["CVE-2024-0001"]})
        with pytest.raises(KeyError):
            inspect_nested(ds, "nonexistent", "id")


# ============================================================================
# Tests: find()
# ============================================================================


class TestFind:
    def test_filter_by_severity(self, sample_df: pd.DataFrame) -> None:
        """find() filters by severity."""
        result = find(sample_df, severity="HIGH")
        assert len(result) == 2

    def test_filter_by_min_score(self, sample_df: pd.DataFrame) -> None:
        """find() filters by minimum CVSS score."""
        result = find(sample_df, min_score=8.0)
        assert len(result) == 2  # 9.8 and 8.1

    def test_filter_by_year(self, sample_df: pd.DataFrame) -> None:
        """find() filters by CVE year."""
        result = find(sample_df, year=2023)
        assert len(result) == 2

    def test_filter_by_status(self, sample_df: pd.DataFrame) -> None:
        """find() filters by NVD status."""
        result = find(sample_df, status="analyzed")
        assert len(result) == 3

    def test_filter_by_keyword(self, sample_df: pd.DataFrame) -> None:
        """find() filters by keyword in description."""
        result = find(sample_df, keyword="injection")
        assert len(result) == 1
        assert result.iloc[0]["cve_id"] == "CVE-2023-0001"

    def test_filter_by_cwe(self, sample_df: pd.DataFrame) -> None:
        """find() filters by CWE ID."""
        result = find(sample_df, cwe="CWE-89")
        assert len(result) == 2

    def test_combined_filters(self, sample_df: pd.DataFrame) -> None:
        """find() combines multiple filters with AND logic."""
        result = find(sample_df, severity="HIGH", year=2024)
        assert len(result) == 1
        assert result.iloc[0]["cve_id"] == "CVE-2024-0002"

    def test_no_filters_returns_all(self, sample_df: pd.DataFrame) -> None:
        """find() with no filters returns all rows."""
        result = find(sample_df)
        assert len(result) == len(sample_df)

    def test_keyword_case_insensitive(self, sample_df: pd.DataFrame) -> None:
        """find() keyword search is case-insensitive."""
        result = find(sample_df, keyword="SQL")
        assert len(result) == 1


# ============================================================================
# Tests: helper functions
# ============================================================================


class TestParseJsonCell:
    def test_none_returns_none(self) -> None:
        assert _parse_json_cell(None) is None

    def test_empty_string_returns_none(self) -> None:
        assert _parse_json_cell("") is None

    def test_null_string_returns_none(self) -> None:
        assert _parse_json_cell("null") is None

    def test_empty_list_string_returns_none(self) -> None:
        assert _parse_json_cell("[]") is None

    def test_valid_json_list(self) -> None:
        result = _parse_json_cell('[{"id": "CWE-89"}]')
        assert result == [{"id": "CWE-89"}]

    def test_already_parsed_dict(self) -> None:
        d = {"id": "CWE-89"}
        assert _parse_json_cell(d) == d

    def test_already_parsed_list(self) -> None:
        lst = [{"id": "CWE-89"}]
        assert _parse_json_cell(lst) == lst

    def test_invalid_json_returns_none(self) -> None:
        assert _parse_json_cell("not json{") is None


class TestExtractField:
    def test_extract_from_dict(self) -> None:
        assert _extract_field({"id": "CWE-89"}, "id") == "CWE-89"

    def test_extract_missing_key_returns_none(self) -> None:
        assert _extract_field({"id": "CWE-89"}, "name") is None

    def test_no_field_returns_stringified(self) -> None:
        result = _extract_field({"id": "CWE-89"}, None)
        assert isinstance(result, str)


# ============================================================================
# Tests: LLM-friendly functions
# ============================================================================


class TestSummary:
    def test_returns_dict(self, sample_dataset: VulnDataset) -> None:
        """summary() returns a JSON-serializable dict."""
        result = summary(sample_dataset)
        assert isinstance(result, dict)
        # to_pandas() filters out rejected status, so 4 not 5
        assert result["record_count"] == 4

    def test_contains_severity(self, sample_dataset: VulnDataset) -> None:
        """summary() includes severity breakdown."""
        result = summary(sample_dataset)
        assert "severity" in result
        assert result["severity"]["HIGH"] == 2

    def test_contains_years(self, sample_dataset: VulnDataset) -> None:
        """summary() includes yearly counts."""
        result = summary(sample_dataset)
        assert "years" in result
        assert result["years"][2023] == 2
        # Rejected CVE-2024-0003 filtered out by to_pandas()
        assert result["years"][2024] == 2

    def test_json_serializable(self, sample_dataset: VulnDataset) -> None:
        """summary() output is JSON-serializable."""
        result = summary(sample_dataset)
        serialized = json.dumps(result, default=str)
        assert isinstance(serialized, str)


class TestDescribeSeverity:
    def test_returns_overall_and_by_year(
        self, sample_dataset: VulnDataset
    ) -> None:
        """describe_severity returns overall and per-year breakdown."""
        result = describe_severity(sample_dataset)
        # to_pandas() filters out rejected status
        assert result["total"] == 4
        assert result["overall"]["HIGH"] == 2
        assert "2023" in result["by_year"]
        assert "2024" in result["by_year"]

    def test_json_serializable(self, sample_dataset: VulnDataset) -> None:
        """describe_severity output is JSON-serializable."""
        result = describe_severity(sample_dataset)
        serialized = json.dumps(result, default=str)
        assert isinstance(serialized, str)


class TestDescribeYearly:
    def test_returns_dict(self, sample_dataset: VulnDataset) -> None:
        """describe_yearly returns year-to-count dict."""
        result = describe_yearly(sample_dataset)
        assert result["2023"] == 2
        # Rejected CVE-2024-0003 filtered out by to_pandas()
        assert result["2024"] == 2

    def test_keys_are_strings(self, sample_dataset: VulnDataset) -> None:
        """describe_yearly uses string keys for JSON compatibility."""
        result = describe_yearly(sample_dataset)
        for key in result:
            assert isinstance(key, str)


class TestDescribeMonthly:
    def test_returns_dict(self, sample_dataset: VulnDataset) -> None:
        """describe_monthly returns month-to-count dict."""
        result = describe_monthly(sample_dataset)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_year_filter(self, sample_dataset: VulnDataset) -> None:
        """describe_monthly filters by year."""
        result = describe_monthly(sample_dataset, year=2024)
        total = sum(result.values())
        # Rejected CVE-2024-0003 filtered out by to_pandas()
        assert total == 2


class TestDescribeNested:
    def test_returns_dict(self, nested_dataset: VulnDataset) -> None:
        """describe_nested returns value-to-count dict."""
        result = describe_nested(nested_dataset, "cwes", "id")
        assert isinstance(result, dict)
        assert result["CWE-89"] == 2

    def test_json_serializable(self, nested_dataset: VulnDataset) -> None:
        """describe_nested output is JSON-serializable."""
        result = describe_nested(nested_dataset, "cwes", "id")
        serialized = json.dumps(result)
        assert isinstance(serialized, str)


class TestSearch:
    def test_returns_list_of_dicts(
        self, sample_dataset: VulnDataset
    ) -> None:
        """search() returns a list of dicts."""
        results = search(sample_dataset, severity="HIGH")
        assert isinstance(results, list)
        assert len(results) == 2
        assert isinstance(results[0], dict)
        assert "cve_id" in results[0]

    def test_limit_parameter(self, sample_dataset: VulnDataset) -> None:
        """search() respects the limit parameter."""
        results = search(sample_dataset, limit=1)
        assert len(results) == 1

    def test_combined_filters(self, sample_dataset: VulnDataset) -> None:
        """search() combines filters."""
        results = search(sample_dataset, severity="HIGH", year=2024)
        assert len(results) == 1
        assert results[0]["cve_id"] == "CVE-2024-0002"

    def test_json_serializable(self, sample_dataset: VulnDataset) -> None:
        """search() results are JSON-serializable."""
        results = search(sample_dataset, severity="HIGH")
        serialized = json.dumps(results, default=str)
        assert isinstance(serialized, str)


class TestLookup:
    def test_returns_dict_for_existing_cve(
        self, sample_dataset: VulnDataset
    ) -> None:
        """lookup() returns dict for existing CVE."""
        result = lookup(sample_dataset, "CVE-2023-0001")
        assert result is not None
        assert result["cve_id"] == "CVE-2023-0001"

    def test_returns_none_for_missing_cve(
        self, sample_dataset: VulnDataset
    ) -> None:
        """lookup() returns None for non-existent CVE."""
        result = lookup(sample_dataset, "CVE-9999-0001")
        assert result is None

    def test_json_serializable(self, sample_dataset: VulnDataset) -> None:
        """lookup() result is JSON-serializable."""
        result = lookup(sample_dataset, "CVE-2023-0001")
        assert result is not None
        serialized = json.dumps(result, default=str)
        assert isinstance(serialized, str)


class TestDataInfo:
    def test_no_cache_reports_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """data_info() reports missing cache."""
        import vulnflow.helpers.explore as explore_mod

        monkeypatch.setattr(explore_mod, "DEFAULT_CACHE_PATH", Path("nonexistent.parquet"))
        result = data_info()
        assert result["has_cache"] is False
        assert result["record_count"] == 0
        assert result["needs_update"] is True

    def test_with_cache_reports_info(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """data_info() reports cache info when available."""
        import vulnflow.helpers.explore as explore_mod

        ds = _make_dataset({"cve_id": ["CVE-2024-0001", "CVE-2024-0002"]})
        cache_path = tmp_path / "test_cache.parquet"
        ds.to_parquet(cache_path)

        monkeypatch.setattr(explore_mod, "DEFAULT_CACHE_PATH", cache_path)
        result = data_info()
        assert result["has_cache"] is True
        assert result["record_count"] == 2
        assert 2024 in result["years"]
