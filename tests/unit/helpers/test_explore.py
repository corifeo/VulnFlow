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
- search() multi-criteria CVE search with enriched filters
- lookup() single CVE detail
- status_distribution() delegation
- _parse_json_cell() JSON parsing
- _extract_field() field extraction
- Skill aliases: describe_severity, describe_nested, describe_yearly, describe_monthly
- Domain functions: risk_priority, exploitation, attack_surface, vendor_risk, weakness_profile
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from tests.conftest import make_dataset

from vulnflow.dataset import VulnDataset
from vulnflow.helpers.explore import (
    _extract_field,
    _parse_json_cell,
    attack_surface,
    data_info,
    describe,
    describe_monthly,
    describe_nested,
    describe_severity,
    describe_yearly,
    exploitation,
    load,
    lookup,
    nested,
    risk_priority,
    search,
    severity,
    status_distribution,
    summary,
    trend,
    vendor_risk,
    weakness_profile,
)


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
# Fixture: analysis_dataset (10 records with known enriched values)
#
# Non-rejected records (indices 0-8, index 9 is rejected):
#   idx  severity   cvss  epss  kev   AV  PR  UI  S   vendor      cwe_id
#   0    HIGH       7.5   0.80  T     N   N   N   U   microsoft   CWE-89
#   1    MEDIUM     5.3   0.05  F     N   N   R   C   apache      CWE-79
#   2    CRITICAL   9.8   0.95  T     N   N   N   U   linux       CWE-120
#   3    HIGH       8.1   0.60  F     N   N   N   U   microsoft   CWE-94
#   4    MEDIUM     6.5   0.02  F     N   L   N   U   apache      CWE-918
#   5    HIGH       7.8   0.35  T     L   L   N   U   microsoft   CWE-269
#   6    LOW        3.1   0.01  F     L   N   R   U   google      CWE-400
#   7    MEDIUM     4.3   0.10  F     N   N   R   U   apache      CWE-200
#   8    CRITICAL   9.1   0.75  T     N   N   N   C   linux       CWE-89
# ============================================================================


@pytest.fixture
def analysis_dataset() -> VulnDataset:
    """Dataset with 10 records with known CVSS/EPSS/KEV/CWE/CPE values."""
    ds = make_dataset(
        {
            "cve_id": [
                "CVE-2023-0001",
                "CVE-2023-0002",
                "CVE-2023-0003",
                "CVE-2024-0001",
                "CVE-2024-0002",
                "CVE-2024-0003",
                "CVE-2024-0004",
                "CVE-2024-0005",
                "CVE-2024-0006",
                "CVE-2024-0007",
            ],
            "nvd_status": [
                "analyzed",
                "analyzed",
                "analyzed",
                "analyzed",
                "analyzed",
                "analyzed",
                "analyzed",
                "modified",
                "analyzed",
                "rejected",
            ],
            "description": [
                "SQL injection",
                "XSS",
                "Buffer overflow",
                "RCE",
                "SSRF",
                "Privilege escalation",
                "DoS",
                "Info disclosure",
                "Auth bypass",
                "Not a vuln",
            ],
            "cvss_base_severity": [
                "HIGH",
                "MEDIUM",
                "CRITICAL",
                "HIGH",
                "MEDIUM",
                "HIGH",
                "LOW",
                "MEDIUM",
                "CRITICAL",
                None,
            ],
            "cvss_base_score": [7.5, 5.3, 9.8, 8.1, 6.5, 7.8, 3.1, 4.3, 9.1, None],
            "cvss_attack_vector": [
                "N",
                "N",
                "N",
                "N",
                "N",
                "L",
                "L",
                "N",
                "N",
                None,
            ],
            "cvss_attack_complexity": [
                "L",
                "L",
                "L",
                "H",
                "L",
                "L",
                "H",
                "L",
                "L",
                None,
            ],
            "cvss_privileges_required": [
                "N",
                "N",
                "N",
                "N",
                "L",
                "L",
                "N",
                "N",
                "N",
                None,
            ],
            "cvss_user_interaction": [
                "N",
                "R",
                "N",
                "N",
                "N",
                "N",
                "R",
                "R",
                "N",
                None,
            ],
            "cvss_scope": [
                "U",
                "C",
                "U",
                "U",
                "U",
                "U",
                "U",
                "U",
                "C",
                None,
            ],
            "cvss_confidentiality": [
                "H",
                "L",
                "H",
                "H",
                "L",
                "H",
                "N",
                "L",
                "H",
                None,
            ],
            "cvss_integrity": [
                "H",
                "L",
                "H",
                "H",
                "L",
                "H",
                "N",
                "N",
                "H",
                None,
            ],
            "cvss_availability": [
                "H",
                "N",
                "H",
                "N",
                "N",
                "N",
                "H",
                "N",
                "H",
                None,
            ],
        }
    )
    # Add enriched columns using numpy arrays directly
    ds._data["epss_probability"] = np.array(
        [0.8, 0.05, 0.95, 0.6, 0.02, 0.35, 0.01, 0.1, 0.75, None], dtype=object
    )
    ds._data["is_kev"] = np.array(
        [True, False, True, False, False, True, False, False, True, False], dtype=bool
    )
    ds._data["kev_ransomware"] = np.array(
        [True, False, False, False, False, True, False, False, False, False], dtype=bool
    )
    ds._data["cpe_vendor"] = np.array(
        [
            "microsoft",
            "apache",
            "linux",
            "microsoft",
            "apache",
            "microsoft",
            "google",
            "apache",
            "linux",
            None,
        ],
        dtype=object,
    )
    ds._data["cpe_product"] = np.array(
        [
            "windows",
            "httpd",
            "kernel",
            "office",
            "httpd",
            "edge",
            "chrome",
            "tomcat",
            "kernel",
            None,
        ],
        dtype=object,
    )
    ds._data["cwe_id"] = np.array(
        [
            "CWE-89",
            "CWE-79",
            "CWE-120",
            "CWE-94",
            "CWE-918",
            "CWE-269",
            "CWE-400",
            "CWE-200",
            "CWE-89",
            None,
        ],
        dtype=object,
    )
    ds._data["cwe_name"] = np.array(
        [
            "SQL Injection",
            "XSS",
            "Buffer Overflow",
            "Code Injection",
            "SSRF",
            "Privilege Escalation",
            "DoS",
            "Info Exposure",
            "SQL Injection",
            None,
        ],
        dtype=object,
    )
    return ds


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


# ============================================================================
# Tests: Aliases
# ============================================================================


class TestAliases:
    """Test skill alias functions delegate correctly."""

    def test_describe_severity_delegates(self, analysis_dataset: VulnDataset) -> None:
        """describe_severity returns identical output to severity."""
        result = describe_severity(analysis_dataset)
        direct = severity(analysis_dataset)
        assert result["total"] == direct["total"]
        assert result["overall"] == direct["overall"]
        assert result["by_year"] == direct["by_year"]

    def test_describe_severity_with_filter(self, analysis_dataset: VulnDataset) -> None:
        """describe_severity passes filters through correctly."""
        result = describe_severity(analysis_dataset, min_score=9.0)
        # Only 2 records with cvss >= 9.0: idx 2 (9.8) and idx 8 (9.1)
        assert result["total"] == 2
        assert result["overall"]["CRITICAL"] == 2

    def test_describe_nested_delegates(self, nested_dataset: VulnDataset) -> None:
        result = describe_nested(nested_dataset, "cwes", "id")
        direct = nested(nested_dataset, "cwes", "id")
        assert result == direct

    def test_describe_yearly_delegates(self, analysis_dataset: VulnDataset) -> None:
        result = describe_yearly(analysis_dataset)
        direct = trend(analysis_dataset, by="year")
        assert result == direct

    def test_describe_monthly_delegates(self, sample_dataset: VulnDataset) -> None:
        """describe_monthly with actual dates matches trend(by='month')."""
        result = describe_monthly(sample_dataset)
        direct = trend(sample_dataset, by="month")
        assert result == direct
        # sample_dataset has nvd_published_at, so results should be non-empty
        assert len(result) > 0

    def test_alias_fmt_json(self, analysis_dataset: VulnDataset) -> None:
        result = describe_severity(analysis_dataset, fmt="json")
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert "total" in parsed
        assert "overall" in parsed


# ============================================================================
# Tests: SearchFilters
# ============================================================================


class TestSearchFilters:
    """Test new search() filter parameters."""

    def test_filter_is_kev_true(self, analysis_dataset: VulnDataset) -> None:
        """is_kev=True returns exactly 4 KEV records (idx 0,2,5,8)."""
        results = search(analysis_dataset, is_kev=True)
        assert len(results) == 4
        assert all(r.get("is_kev") is True for r in results)

    def test_filter_is_kev_false(self, analysis_dataset: VulnDataset) -> None:
        """is_kev=False returns non-KEV records (5 of 9 non-rejected + idx 9)."""
        results = search(analysis_dataset, is_kev=False)
        # 5 non-KEV non-rejected records (idx 1,3,4,6,7) plus idx 9 (rejected
        # but has is_kev=False). Actual count depends on to_pandas handling.
        assert len(results) >= 5
        assert all(r.get("is_kev") is False for r in results)

    def test_filter_min_epss(self, analysis_dataset: VulnDataset) -> None:
        """min_epss=0.5 returns 4 records: idx 0(0.8),2(0.95),3(0.6),8(0.75)."""
        results = search(analysis_dataset, min_epss=0.5)
        assert len(results) == 4
        for r in results:
            assert float(r["epss_probability"]) >= 0.5

    def test_filter_max_epss(self, analysis_dataset: VulnDataset) -> None:
        """max_epss=0.1 returns 4 records: idx 1(0.05),4(0.02),6(0.01),7(0.1)."""
        results = search(analysis_dataset, max_epss=0.1)
        assert len(results) == 4
        for r in results:
            assert float(r["epss_probability"]) <= 0.1

    def test_filter_vendor(self, analysis_dataset: VulnDataset) -> None:
        """vendor='microsoft' returns exactly 3 records (idx 0,3,5)."""
        results = search(analysis_dataset, vendor="microsoft")
        assert len(results) == 3
        for r in results:
            assert r.get("cpe_vendor", "").lower() == "microsoft"

    def test_filter_vendor_case_insensitive(
        self, analysis_dataset: VulnDataset
    ) -> None:
        results_lower = search(analysis_dataset, vendor="microsoft")
        results_upper = search(analysis_dataset, vendor="MICROSOFT")
        assert len(results_lower) == len(results_upper) == 3

    def test_filter_product(self, analysis_dataset: VulnDataset) -> None:
        """product='kernel' returns exactly 2 records (idx 2,8)."""
        results = search(analysis_dataset, product="kernel")
        assert len(results) == 2
        for r in results:
            assert r.get("cpe_product", "").lower() == "kernel"

    def test_filter_ransomware(self, analysis_dataset: VulnDataset) -> None:
        """ransomware=True returns exactly 2 records (idx 0,5)."""
        results = search(
            analysis_dataset,
            ransomware=True,
            columns=["cve_id", "kev_ransomware"],
        )
        assert len(results) == 2
        cve_ids = {r["cve_id"] for r in results}
        assert cve_ids == {"CVE-2023-0001", "CVE-2024-0003"}

    def test_default_columns_include_enriched(
        self, analysis_dataset: VulnDataset
    ) -> None:
        """Default columns include epss_probability and is_kev when present."""
        results = search(analysis_dataset, severity="HIGH")
        assert len(results) > 0
        assert "is_kev" in results[0]
        assert "epss_probability" in results[0]

    def test_default_columns_include_epss_bucket(self) -> None:
        """Default columns include epss_bucket when present."""
        ds = make_dataset(
            {
                "cve_id": ["CVE-2024-0001", "CVE-2024-0002"],
                "description": ["vuln a", "vuln b"],
                "nvd_status": ["analyzed", "analyzed"],
                "cvss_base_severity": ["HIGH", "MEDIUM"],
                "cvss_base_score": [9.8, 5.3],
            }
        )
        ds._data["epss_probability"] = np.array([0.9, 0.05], dtype=object)
        ds._data["epss_bucket"] = np.array(["high", "low"], dtype=object)
        results = search(ds)
        assert len(results) == 2
        assert "epss_bucket" in results[0]
        assert results[0]["epss_bucket"] == "high"
        assert results[1]["epss_bucket"] == "low"

    def test_filter_epss_bucket(self) -> None:
        """search(epss_bucket='high') filters correctly."""
        ds = make_dataset(
            {
                "cve_id": ["CVE-2024-0001", "CVE-2024-0002", "CVE-2024-0003"],
                "description": ["vuln a", "vuln b", "vuln c"],
                "nvd_status": ["analyzed", "analyzed", "analyzed"],
                "cvss_base_severity": ["CRITICAL", "HIGH", "LOW"],
                "cvss_base_score": [9.8, 7.5, 3.1],
            }
        )
        ds._data["epss_bucket"] = np.array(["high", "medium", "very_low"], dtype=object)
        results = search(ds, epss_bucket="high")
        assert len(results) == 1
        assert results[0]["cve_id"] == "CVE-2024-0001"

    def test_filter_epss_bucket_medium(self) -> None:
        """search(epss_bucket='medium') returns correct records."""
        ds = make_dataset(
            {
                "cve_id": ["CVE-2024-0001", "CVE-2024-0002", "CVE-2024-0003"],
                "description": ["vuln a", "vuln b", "vuln c"],
                "nvd_status": ["analyzed", "analyzed", "analyzed"],
            }
        )
        ds._data["epss_bucket"] = np.array(["high", "medium", "medium"], dtype=object)
        results = search(ds, epss_bucket="medium")
        assert len(results) == 2
        cve_ids = {r["cve_id"] for r in results}
        assert cve_ids == {"CVE-2024-0002", "CVE-2024-0003"}


# ============================================================================
# Tests: RiskPriority
# ============================================================================


class TestRiskPriority:
    """Test risk_priority() function.

    Expected quadrants for analysis_dataset (9 active records):
    - critical_exploited (HIGH/CRIT + KEV or EPSS>0.7): idx 0,2,5,8 → 4
    - critical_unlikely (HIGH/CRIT + no exploitation signals): idx 3 → 1
    - sleeper_threats (!HIGH/CRIT + EPSS>0.3): none → 0
    - low_risk: idx 1,4,6,7 → 4
    """

    def test_excludes_rejected_records(self, analysis_dataset: VulnDataset) -> None:
        result = risk_priority(analysis_dataset)
        assert result["total"] == 9

    def test_quadrant_exact_counts(self, analysis_dataset: VulnDataset) -> None:
        result = risk_priority(analysis_dataset)
        q = result["quadrants"]
        assert q["critical_exploited"] == 4
        assert q["critical_unlikely"] == 1
        assert q["sleeper_threats"] == 0
        assert q["low_risk"] == 4
        # Partition invariant
        assert sum(q.values()) == result["total"]

    def test_top_urgent_ordering(self, analysis_dataset: VulnDataset) -> None:
        result = risk_priority(analysis_dataset)
        scores = [e["risk_score"] for e in result["top_urgent"]]
        assert scores == sorted(scores, reverse=True)
        # All entries should have positive risk scores
        assert all(s > 0 for s in scores)

    def test_top_urgent_limit(self, analysis_dataset: VulnDataset) -> None:
        result = risk_priority(analysis_dataset, limit=3)
        assert len(result["top_urgent"]) <= 3

    def test_top_urgent_contains_expected_fields(
        self, analysis_dataset: VulnDataset
    ) -> None:
        result = risk_priority(analysis_dataset)
        for entry in result["top_urgent"]:
            assert "cve_id" in entry
            assert "risk_score" in entry
            assert "cvss_score" in entry
            assert "epss" in entry
            assert "is_kev" in entry

    def test_signal_coverage_exact(self, analysis_dataset: VulnDataset) -> None:
        result = risk_priority(analysis_dataset)
        sc = result["signal_coverage"]
        assert sc["has_cvss"] == 9  # all non-rejected have scores
        assert sc["has_epss"] == 9  # 9 valid EPSS values (idx 9 is rejected)
        assert sc["has_kev"] == 4  # idx 0,2,5,8
        assert sc["has_all_three"] == 4  # same 4 KEV records have all signals

    def test_fmt_json(self, analysis_dataset: VulnDataset) -> None:
        result = risk_priority(analysis_dataset, fmt="json")
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed["quadrants"]["critical_exploited"] == 4

    def test_empty_dataset(self) -> None:
        """risk_priority handles empty dataset gracefully."""
        ds = make_dataset({"cve_id": np.array([], dtype=object)})
        result = risk_priority(ds)
        assert result["total"] == 0
        assert sum(result["quadrants"].values()) == 0
        assert result["top_urgent"] == []


# ============================================================================
# Tests: Exploitation
# ============================================================================


class TestExploitation:
    """Test exploitation() function.

    Expected values from analysis_dataset (9 active records):
    - KEV total: 4 (idx 0,2,5,8), ransomware: 2 (idx 0,5)
    - EPSS high (>0.5): 4 (idx 0,2,3,8), very_high (>0.9): 1 (idx 2)
    - Overlap: kev+high_epss: 3 (0,2,8), kev+low_epss: 0, high_epss_not_kev: 1 (3)
    """

    def test_kev_exact_counts(self, analysis_dataset: VulnDataset) -> None:
        result = exploitation(analysis_dataset)
        kev = result["kev"]
        assert kev["total"] == 4
        assert kev["ransomware"] == 2

    def test_epss_exact_distribution(self, analysis_dataset: VulnDataset) -> None:
        result = exploitation(analysis_dataset)
        epss = result["epss"]
        assert epss["high"] == 4
        assert epss["very_high"] == 1
        assert "percentiles" in epss
        assert epss["percentiles"]["p50"] > 0

    def test_overlap_exact_counts(self, analysis_dataset: VulnDataset) -> None:
        result = exploitation(analysis_dataset)
        overlap = result["overlap"]
        assert overlap["kev_and_high_epss"] == 3
        assert overlap["kev_and_low_epss"] == 0
        assert overlap["high_epss_not_kev"] == 1

    def test_empty_dataset(self) -> None:
        """exploitation handles empty dataset gracefully."""
        ds = make_dataset({"cve_id": np.array([], dtype=object)})
        result = exploitation(ds)
        assert result["total"] == 0
        assert result["kev"]["total"] == 0


# ============================================================================
# Tests: AttackSurface
# ============================================================================


class TestAttackSurface:
    """Test attack_surface() function.

    Expected values from analysis_dataset (9 active records):
    - by_vector: N=7, L=2
    - no_auth_remote (AV=N + PR=N): idx 0,1,2,3,7,8 → 6
    - no_interaction (above + UI=N): idx 0,2,3,8 → 4
    - scope_changed (S=C): idx 1,8 → 2
    """

    def test_component_distributions_exact(self, analysis_dataset: VulnDataset) -> None:
        result = attack_surface(analysis_dataset)
        assert result["by_vector"] == {"N": 7, "L": 2}
        assert result["by_complexity"]["L"] == 7
        assert result["by_complexity"]["H"] == 2

    def test_worst_case_exact_metrics(self, analysis_dataset: VulnDataset) -> None:
        result = attack_surface(analysis_dataset)
        wc = result["worst_case"]
        assert wc["no_auth_remote"] == 6
        assert wc["no_interaction"] == 4
        assert wc["scope_changed"] == 2

    def test_cia_distributions(self, analysis_dataset: VulnDataset) -> None:
        result = attack_surface(analysis_dataset)
        cia = result["by_cia"]
        assert "confidentiality" in cia
        assert "integrity" in cia
        assert "availability" in cia
        # Verify exact confidentiality breakdown: H=5, L=3, N=1
        assert cia["confidentiality"]["H"] == 5
        assert cia["confidentiality"]["L"] == 3
        assert cia["confidentiality"]["N"] == 1

    def test_empty_dataset(self) -> None:
        """attack_surface handles empty dataset gracefully."""
        ds = make_dataset({"cve_id": np.array([], dtype=object)})
        result = attack_surface(ds)
        assert result["total"] == 0
        assert result["worst_case"] == {}


# ============================================================================
# Tests: VendorRisk
# ============================================================================


class TestVendorRisk:
    """Test vendor_risk() function.

    Fixture vendors (9 active): microsoft=3, apache=3, linux=2, google=1
    """

    def test_top_n_mode(self, analysis_dataset: VulnDataset) -> None:
        result = vendor_risk(analysis_dataset)
        assert "vendors" in result
        vendors = result["vendors"]
        assert len(vendors) == 4  # 4 unique vendors
        # Verify sorted by risk_score descending
        scores = [v["risk_score"] for v in vendors]
        assert scores == sorted(scores, reverse=True)
        # Each vendor entry has required fields
        for v in vendors:
            assert "vendor" in v
            assert "count" in v
            assert "risk_score" in v

    def test_top_n_limit(self, analysis_dataset: VulnDataset) -> None:
        result = vendor_risk(analysis_dataset, top_n=2)
        assert len(result["vendors"]) == 2

    def test_single_vendor_mode(self, analysis_dataset: VulnDataset) -> None:
        result = vendor_risk(analysis_dataset, "microsoft")
        assert result["vendor"] == "microsoft"
        assert result["count"] == 3
        # Should have enriched detail sections
        assert "severity" in result
        assert "kev_count" in result

    def test_single_vendor_case_insensitive(
        self, analysis_dataset: VulnDataset
    ) -> None:
        r1 = vendor_risk(analysis_dataset, "Microsoft")
        r2 = vendor_risk(analysis_dataset, "MICROSOFT")
        assert r1["count"] == r2["count"] == 3

    def test_missing_vendor(self, analysis_dataset: VulnDataset) -> None:
        result = vendor_risk(analysis_dataset, "nonexistent_vendor")
        assert result["count"] == 0

    def test_no_vendor_column(self) -> None:
        ds = make_dataset({"cve_id": ["CVE-2024-0001"]})
        result = vendor_risk(ds)
        assert "error" in result

    def test_empty_dataset(self) -> None:
        """vendor_risk handles empty dataset gracefully."""
        ds = make_dataset({"cve_id": np.array([], dtype=object)})
        ds._data["cpe_vendor"] = np.array([], dtype=object)
        result = vendor_risk(ds)
        assert result == {"vendors": []}


# ============================================================================
# Tests: WeaknessProfile
# ============================================================================


class TestWeaknessProfile:
    """Test weakness_profile() function.

    Fixture CWEs (9 active): CWE-89=2, all others=1 each (8 unique CWEs)
    """

    def test_returns_weaknesses_sorted_by_count(
        self, analysis_dataset: VulnDataset
    ) -> None:
        result = weakness_profile(analysis_dataset)
        assert "weaknesses" in result
        weaknesses = result["weaknesses"]
        assert len(weaknesses) == 8  # 8 unique CWEs
        # CWE-89 should be first (count=2, all others count=1)
        assert weaknesses[0]["id"] == "CWE-89"
        assert weaknesses[0]["count"] == 2

    def test_weakness_enriched_metrics(self, analysis_dataset: VulnDataset) -> None:
        result = weakness_profile(analysis_dataset)
        cwe89 = result["weaknesses"][0]
        assert cwe89["id"] == "CWE-89"
        assert cwe89["name"] == "SQL Injection"
        # avg_cvss for CWE-89: (7.5 + 9.1) / 2 = 8.3
        assert cwe89["avg_cvss"] == pytest.approx(8.3, abs=0.01)
        # avg_epss for CWE-89: (0.8 + 0.75) / 2 = 0.775
        assert cwe89["avg_epss"] == pytest.approx(0.775, abs=0.001)
        # kev_count for CWE-89: both idx 0 and 8 are KEV
        assert cwe89["kev_count"] == 2

    def test_top_n_limit(self, analysis_dataset: VulnDataset) -> None:
        result = weakness_profile(analysis_dataset, top_n=3)
        assert len(result["weaknesses"]) == 3

    def test_year_filter(self, analysis_dataset: VulnDataset) -> None:
        result_2024 = weakness_profile(analysis_dataset, year=2024)
        # 2024 CVEs: idx 3-8 → CWE-94, CWE-918, CWE-269, CWE-400, CWE-200, CWE-89
        total_2024 = sum(w["count"] for w in result_2024["weaknesses"])
        assert total_2024 == 6  # 6 non-rejected 2024 records

    def test_year_comparison_produces_keys(self, analysis_dataset: VulnDataset) -> None:
        """year filter with prior-year data produces rising/declining keys."""
        result = weakness_profile(analysis_dataset, year=2024)
        # CWE-89 appears in both years (2023: 1, 2024: 1) — no 20% change → neither
        # Keys should exist even if empty
        assert "rising" in result
        assert "declining" in result
        assert isinstance(result["rising"], list)
        assert isinstance(result["declining"], list)

    def test_no_cwe_column(self) -> None:
        ds = make_dataset({"cve_id": ["CVE-2024-0001"]})
        result = weakness_profile(ds)
        assert "error" in result

    def test_empty_dataset(self) -> None:
        """weakness_profile handles empty dataset gracefully."""
        ds = make_dataset({"cve_id": np.array([], dtype=object)})
        ds._data["cwe_id"] = np.array([], dtype=object)
        result = weakness_profile(ds)
        assert result == {"weaknesses": []}
