"""
Tests for vulnflow.helpers.explore module.

Tests:
- load() cache resolution and file extension dispatch
- data_info() cache status reporting
- describe() adaptive column profiling
- summary() structured overview (numpy-first)
- severity() severity breakdown with filters (numpy-first)
- trend() temporal counts with parameterized granularity
- nested() JSON column top values
- search() multi-criteria CVE search
- lookup() single CVE detail
- status_distribution() delegation
- _parse_json_cell() JSON parsing
- _extract_field() field extraction
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vulnflow.dataset import VulnDataset
from vulnflow.helpers.explore import (
    _extract_field,
    _parse_json_cell,
    data_info,
    describe,
    load,
    lookup,
    nested,
    search,
    severity,
    status_distribution,
    summary,
    trend,
)

from tests.conftest import make_dataset


@pytest.fixture
def sample_dataset() -> VulnDataset:
    """Dataset with basic columns for testing."""
    return make_dataset(
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
    return make_dataset(
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
        ds = make_dataset({"cve_id": ["CVE-2024-0001"]})
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
        ds = make_dataset({"cve_id": ["CVE-2024-0001"]})
        json_path = tmp_path / "test.json"
        ds.to_json(json_path)

        result = load(str(json_path))
        assert len(result) == 1


# ============================================================================
# Tests: data_info()
# ============================================================================


class TestDataInfo:
    def test_no_cache_reports_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """data_info() reports missing cache."""
        import vulnflow.helpers.explore as explore_mod

        monkeypatch.setattr(
            explore_mod, "DEFAULT_CACHE_PATH", Path("nonexistent.parquet")
        )
        result = data_info()
        assert result["has_cache"] is False
        assert result["record_count"] == 0
        assert result["needs_update"] is True

    def test_with_cache_reports_info(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """data_info() reports cache info when available."""
        import vulnflow.helpers.explore as explore_mod

        ds = make_dataset({"cve_id": ["CVE-2024-0001", "CVE-2024-0002"]})
        cache_path = tmp_path / "test_cache.parquet"
        ds.to_parquet(cache_path)

        monkeypatch.setattr(explore_mod, "DEFAULT_CACHE_PATH", cache_path)
        result = data_info()
        assert result["has_cache"] is True
        assert result["record_count"] == 2
        assert 2024 in result["years"]


# ============================================================================
# Tests: describe()
# ============================================================================


class TestDescribe:
    def test_describe_returns_dict(self, sample_dataset: VulnDataset) -> None:
        """describe() returns dict with record_count, column_count, columns."""
        result = describe(sample_dataset)
        assert isinstance(result, dict)
        assert result["record_count"] == 5
        assert result["column_count"] == len(sample_dataset.columns)
        assert isinstance(result["columns"], list)
        assert len(result["columns"]) == len(sample_dataset.columns)

    def test_describe_numeric_column(self) -> None:
        """Numeric columns get min, max, mean, p50 stats."""
        ds = make_dataset(
            {
                "cve_id": ["CVE-2024-0001", "CVE-2024-0002", "CVE-2024-0003"],
                "cvss_base_score": [3.0, 7.0, 10.0],
            }
        )
        result = describe(ds)
        col_info = _find_col(result, "cvss_base_score")
        assert col_info is not None
        assert col_info["kind"] == "numeric"
        assert col_info["min"] == 3.0
        assert col_info["max"] == 10.0
        assert col_info["mean"] == pytest.approx(6.67, abs=0.01)
        assert "p50" in col_info

    def test_describe_categorical_column(self) -> None:
        """Low-cardinality string columns get top_values."""
        ds = make_dataset(
            {
                "cve_id": ["CVE-2024-0001", "CVE-2024-0002", "CVE-2024-0003"],
                "cvss_base_severity": ["HIGH", "HIGH", "MEDIUM"],
            }
        )
        result = describe(ds)
        col_info = _find_col(result, "cvss_base_severity")
        assert col_info is not None
        assert col_info["kind"] == "categorical"
        assert col_info["top_values"]["HIGH"] == 2
        assert col_info["top_values"]["MEDIUM"] == 1

    def test_describe_high_cardinality(self) -> None:
        """High-cardinality columns (>25 unique) get top 5 only by default."""
        values = [f"vendor_{i}" for i in range(30)]
        ds = make_dataset(
            {
                "cve_id": [f"CVE-2024-{i:04d}" for i in range(30)],
                "cpe_vendor": values,
            }
        )
        result = describe(ds, top_n=5)
        col_info = _find_col(result, "cpe_vendor")
        assert col_info is not None
        assert col_info["kind"] == "high_cardinality"
        assert col_info["unique_count"] == 30
        assert len(col_info["top_values"]) <= 5

    def test_describe_timestamp_column(self) -> None:
        """Datetime columns get min and max."""
        ds = make_dataset(
            {
                "cve_id": ["CVE-2024-0001", "CVE-2024-0002"],
            }
        )
        # Create a proper datetime array
        ts = np.array(["2023-01-15", "2024-06-20"], dtype="datetime64[ns]")
        ds._data["nvd_published_at"] = ts
        result = describe(ds)
        col_info = _find_col(result, "nvd_published_at")
        assert col_info is not None
        assert col_info["kind"] == "timestamp"
        assert "min" in col_info
        assert "max" in col_info

    def test_describe_json_array_column(self) -> None:
        """Nested JSON array columns get avg_items."""
        cwes = [
            json.dumps([{"id": "CWE-89"}, {"id": "CWE-79"}]),
            json.dumps([{"id": "CWE-120"}]),
            json.dumps([{"id": "CWE-89"}, {"id": "CWE-79"}, {"id": "CWE-120"}]),
        ]
        ds = make_dataset(
            {
                "cve_id": ["CVE-2024-0001", "CVE-2024-0002", "CVE-2024-0003"],
                "cwes": cwes,
            }
        )
        result = describe(ds)
        col_info = _find_col(result, "cwes")
        assert col_info is not None
        assert col_info["kind"] == "json_array"
        assert col_info["avg_items"] == pytest.approx(2.0)

    def test_describe_bool_column(self) -> None:
        """Bool columns get true_count and false_count."""
        ds = make_dataset(
            {
                "cve_id": ["CVE-2024-0001", "CVE-2024-0002", "CVE-2024-0003"],
            }
        )
        ds._data["is_kev"] = np.array([True, False, True], dtype=bool)
        result = describe(ds)
        col_info = _find_col(result, "is_kev")
        assert col_info is not None
        assert col_info["kind"] == "bool"
        assert col_info["true_count"] == 2
        assert col_info["false_count"] == 1

    def test_describe_fmt_json(self) -> None:
        """describe(fmt='json') returns a valid JSON string."""
        ds = make_dataset(
            {
                "cve_id": ["CVE-2024-0001"],
                "cvss_base_score": [7.5],
            }
        )
        result = describe(ds, fmt="json")
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed["record_count"] == 1

    def test_describe_fill_rate(self) -> None:
        """Null values reduce fill_pct below 100."""
        ds = make_dataset(
            {
                "cve_id": [
                    "CVE-2024-0001",
                    "CVE-2024-0002",
                    "CVE-2024-0003",
                    "CVE-2024-0004",
                ],
                "cvss_base_severity": ["HIGH", None, "MEDIUM", None],
            }
        )
        result = describe(ds)
        col_info = _find_col(result, "cvss_base_severity")
        assert col_info is not None
        assert col_info["fill_pct"] == pytest.approx(50.0)


# ============================================================================
# Tests: summary()
# ============================================================================


class TestSummary:
    def test_returns_dict(self, sample_dataset: VulnDataset) -> None:
        """summary() returns a JSON-serializable dict."""
        result = summary(sample_dataset)
        assert isinstance(result, dict)
        # _analysis_mask excludes rejected, so 4 not 5
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
        # Rejected CVE-2024-0003 filtered out by _analysis_mask
        assert result["years"][2024] == 2

    def test_json_serializable(self, sample_dataset: VulnDataset) -> None:
        """summary() output is JSON-serializable."""
        result = summary(sample_dataset)
        serialized = json.dumps(result, default=str)
        assert isinstance(serialized, str)


# ============================================================================
# Tests: severity()
# ============================================================================


class TestSeverity:
    def test_returns_overall_and_by_year(self, sample_dataset: VulnDataset) -> None:
        """severity() returns overall and per-year breakdown."""
        result = severity(sample_dataset)
        # _analysis_mask excludes rejected
        assert result["total"] == 4
        assert result["overall"]["HIGH"] == 2
        assert "2023" in result["by_year"]
        assert "2024" in result["by_year"]

    def test_json_serializable(self, sample_dataset: VulnDataset) -> None:
        """severity() output is JSON-serializable."""
        result = severity(sample_dataset)
        serialized = json.dumps(result, default=str)
        assert isinstance(serialized, str)

    def test_filter_by_attack_vector(self) -> None:
        """severity(attack_vector='NETWORK') returns only network CVEs."""
        ds = make_dataset(
            {
                "cve_id": ["CVE-2024-0001", "CVE-2024-0002", "CVE-2024-0003"],
                "cvss_base_severity": ["CRITICAL", "HIGH", "MEDIUM"],
                "cvss_base_score": [9.8, 7.5, 5.3],
                "cvss_attack_vector": ["N", "N", "L"],
                "nvd_status": ["analyzed", "analyzed", "analyzed"],
            }
        )
        result = severity(ds, attack_vector="NETWORK")
        assert result["total"] == 2
        assert "CRITICAL" in result["overall"]
        assert "HIGH" in result["overall"]
        assert "MEDIUM" not in result["overall"]

    def test_filter_by_min_score(self) -> None:
        """severity(min_score=9.0) returns only high-score CVEs."""
        ds = make_dataset(
            {
                "cve_id": ["CVE-2024-0001", "CVE-2024-0002", "CVE-2024-0003"],
                "cvss_base_severity": ["CRITICAL", "HIGH", "MEDIUM"],
                "cvss_base_score": [9.8, 7.5, 5.3],
                "nvd_status": ["analyzed", "analyzed", "analyzed"],
            }
        )
        result = severity(ds, min_score=9.0)
        assert result["total"] == 1
        assert result["overall"]["CRITICAL"] == 1
        assert "HIGH" not in result["overall"]

    def test_combined_attack_vector_and_min_score(self) -> None:
        """severity() with both filters uses AND logic."""
        ds = make_dataset(
            {
                "cve_id": ["CVE-2024-0001", "CVE-2024-0002", "CVE-2024-0003"],
                "cvss_base_severity": ["CRITICAL", "HIGH", "CRITICAL"],
                "cvss_base_score": [9.8, 7.5, 9.1],
                "cvss_attack_vector": ["N", "N", "L"],
                "nvd_status": ["analyzed", "analyzed", "analyzed"],
            }
        )
        result = severity(ds, attack_vector="NETWORK", min_score=9.0)
        assert result["total"] == 1
        assert result["overall"]["CRITICAL"] == 1

    def test_no_filter_regression(self, sample_dataset: VulnDataset) -> None:
        """severity() with no filters behaves the same as before."""
        result = severity(sample_dataset)
        # _analysis_mask excludes rejected -> 4 records
        assert result["total"] == 4
        assert result["overall"]["HIGH"] == 2


# ============================================================================
# Tests: trend()
# ============================================================================


class TestTrend:
    def test_yearly_returns_dict(self, sample_dataset: VulnDataset) -> None:
        """trend(by='year') returns year-to-count dict."""
        result = trend(sample_dataset)
        assert result["2023"] == 2
        # Rejected CVE-2024-0003 filtered out by _analysis_mask
        assert result["2024"] == 2

    def test_yearly_keys_are_strings(self, sample_dataset: VulnDataset) -> None:
        """trend() uses string keys for JSON compatibility."""
        result = trend(sample_dataset)
        for key in result:
            assert isinstance(key, str)

    def test_monthly_returns_dict(self, sample_dataset: VulnDataset) -> None:
        """trend(by='month') returns month-to-count dict."""
        result = trend(sample_dataset, by="month")
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_monthly_year_filter(self, sample_dataset: VulnDataset) -> None:
        """trend(by='month', year=2024) filters by year."""
        result = trend(sample_dataset, by="month", year=2024)
        total = sum(result.values())
        # Rejected CVE-2024-0003 filtered out by _analysis_mask
        assert total == 2

    def test_monthly_key_format(self, sample_dataset: VulnDataset) -> None:
        """trend(by='month') keys are YYYY-MM format strings."""
        result = trend(sample_dataset, by="month")
        for key in result:
            assert isinstance(key, str)
            # Should match YYYY-MM pattern
            parts = key.split("-")
            assert len(parts) == 2
            assert len(parts[0]) == 4
            assert len(parts[1]) == 2


# ============================================================================
# Tests: nested()
# ============================================================================


class TestNested:
    def test_returns_dict(self, nested_dataset: VulnDataset) -> None:
        """nested() returns value-to-count dict."""
        result = nested(nested_dataset, "cwes", "id")
        assert isinstance(result, dict)
        assert result["CWE-89"] == 2

    def test_json_serializable(self, nested_dataset: VulnDataset) -> None:
        """nested() output is JSON-serializable."""
        result = nested(nested_dataset, "cwes", "id")
        serialized = json.dumps(result)
        assert isinstance(serialized, str)


# ============================================================================
# Tests: search()
# ============================================================================


class TestSearch:
    def test_returns_list_of_dicts(self, sample_dataset: VulnDataset) -> None:
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

    def test_filter_by_keyword(self, sample_dataset: VulnDataset) -> None:
        """search() filters by keyword in description."""
        results = search(sample_dataset, keyword="injection")
        assert len(results) == 1
        assert results[0]["cve_id"] == "CVE-2023-0001"

    def test_filter_by_cwe(self, sample_dataset: VulnDataset) -> None:
        """search() filters by CWE ID."""
        results = search(sample_dataset, cwe="CWE-89")
        assert len(results) == 2

    def test_keyword_case_insensitive(self, sample_dataset: VulnDataset) -> None:
        """search() keyword search is case-insensitive."""
        results = search(sample_dataset, keyword="SQL")
        assert len(results) == 1

    def test_attack_vector_full_name(self) -> None:
        """search() with attack_vector='NETWORK' filters correctly."""
        ds = make_dataset(
            {
                "cve_id": ["CVE-2024-0001", "CVE-2024-0002", "CVE-2024-0003"],
                "cvss_attack_vector": ["N", "L", "N"],
                "cvss_base_severity": ["HIGH", "MEDIUM", "CRITICAL"],
                "cvss_base_score": [7.5, 5.3, 9.8],
                "description": ["vuln a", "vuln b", "vuln c"],
                "nvd_status": ["analyzed", "analyzed", "analyzed"],
            }
        )
        results = search(ds, attack_vector="NETWORK")
        assert len(results) == 2

    def test_attack_vector_abbreviation(self) -> None:
        """search() with attack_vector='N' also works."""
        ds = make_dataset(
            {
                "cve_id": ["CVE-2024-0001", "CVE-2024-0002"],
                "cvss_attack_vector": ["N", "L"],
                "cvss_base_severity": ["HIGH", "MEDIUM"],
                "cvss_base_score": [7.5, 5.3],
                "description": ["vuln a", "vuln b"],
                "nvd_status": ["analyzed", "analyzed"],
            }
        )
        results = search(ds, attack_vector="N")
        assert len(results) == 1


# ============================================================================
# Tests: lookup()
# ============================================================================


class TestLookup:
    def test_returns_dict_for_existing_cve(self, sample_dataset: VulnDataset) -> None:
        """lookup() returns dict for existing CVE."""
        result = lookup(sample_dataset, "CVE-2023-0001")
        assert result is not None
        assert result["cve_id"] == "CVE-2023-0001"

    def test_returns_none_for_missing_cve(self, sample_dataset: VulnDataset) -> None:
        """lookup() returns None for non-existent CVE."""
        result = lookup(sample_dataset, "CVE-9999-0001")
        assert result is None

    def test_json_serializable(self, sample_dataset: VulnDataset) -> None:
        """lookup() result is JSON-serializable."""
        result = lookup(sample_dataset, "CVE-2023-0001")
        assert result is not None
        serialized = json.dumps(result, default=str)
        assert isinstance(serialized, str)


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
# Helpers
# ============================================================================


def _find_col(result: dict[str, Any], name: str) -> dict[str, Any] | None:
    """Find a column entry in describe() output by name."""
    for col in result["columns"]:
        if col["name"] == name:
            return col
    return None
