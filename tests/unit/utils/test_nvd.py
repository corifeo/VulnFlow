"""
NVD Parsing Utilities - parse_timestamp, extract_nvd_status, parse_cvss, etc.

Layer: Utils
Dependencies: schema, utils.cvss, utils.cpe
"""

from __future__ import annotations

from typing import Any

import pytest

from vulnflow.utils.nvd import (
    extract_nvd_status,
    parse_cpes,
    parse_cvss,
    parse_cwes,
    parse_references,
    parse_timestamp,
)

# --- Helpers ---


def _make_nvd_metric(
    version: str = "3.1",
    key_suffix: str = "V31",
    base_score: float = 9.8,
    base_severity: str | None = "CRITICAL",
    source: str = "nvd@nist.gov",
    metric_type: str = "Primary",
    vector_string: str = "",
    severity_at_metric_level: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """Build an NVD metrics dict for a single CVSS entry.

    Args:
        severity_at_metric_level: If True, puts baseSeverity on the metric
            (v2.0 style) instead of in cvssData. If base_severity is None,
            omits it entirely (v4.0 style).
    """
    cvss_data: dict[str, Any] = {
        "version": version,
        "vectorString": vector_string,
        "baseScore": base_score,
    }
    metric: dict[str, Any] = {
        "source": source,
        "type": metric_type,
        "cvssData": cvss_data,
    }

    if base_severity is not None:
        if severity_at_metric_level:
            metric["baseSeverity"] = base_severity
        else:
            cvss_data["baseSeverity"] = base_severity

    return {f"cvssMetric{key_suffix}": [metric]}


# --- parse_timestamp ---


def test_parse_timestamp_valid() -> None:
    """Parse ISO timestamp with Z suffix."""
    result = parse_timestamp("2024-01-10T10:00:00.000Z")
    assert result is not None
    assert result.year == 2024
    assert result.month == 1
    assert result.day == 10


def test_parse_timestamp_none() -> None:
    """None and empty string return None."""
    assert parse_timestamp(None) is None
    assert parse_timestamp("") is None


# --- extract_nvd_status ---


@pytest.mark.parametrize(
    ("vuln_status", "expected"),
    [
        ("Analyzed", "analyzed"),
        ("Modified", "modified"),
        ("Rejected", "rejected"),
        ("Deferred", "deferred"),
    ],
)
def test_extract_nvd_status_direct_mapping(vuln_status: str, expected: str) -> None:
    """Direct vulnStatus values map to normalized form."""
    assert extract_nvd_status({"vulnStatus": vuln_status}) == expected


def test_extract_nvd_status_awaiting_variants() -> None:
    """Awaiting, Received, Undergoing all map to 'awaiting'."""
    assert extract_nvd_status({"vulnStatus": "Awaiting Analysis"}) == "awaiting"
    assert extract_nvd_status({"vulnStatus": "Received"}) == "awaiting"
    assert extract_nvd_status({"vulnStatus": "Undergoing Analysis"}) == "awaiting"


def test_extract_nvd_status_missing_defaults_analyzed() -> None:
    """Missing vulnStatus defaults to analyzed."""
    assert extract_nvd_status({"descriptions": [{"value": "Normal vuln"}]}) == "analyzed"


def test_extract_nvd_status_missing_rejected_description() -> None:
    """Missing vulnStatus with REJECTED description -> rejected."""
    cve_data = {"descriptions": [{"value": "** REJECTED ** Duplicate"}]}
    assert extract_nvd_status(cve_data) == "rejected"


# --- parse_cvss ---


def test_parse_cvss_v40_computes_severity() -> None:
    """CVSS 4.0: baseSeverity computed from score (not in JSON)."""
    metrics = _make_nvd_metric(
        version="4.0",
        key_suffix="V40",
        base_score=9.3,
        base_severity=None,  # v4.0 has no baseSeverity in cvssData
    )
    entries = parse_cvss(metrics)
    assert len(entries) == 1
    assert entries[0].version == "4.0"
    assert entries[0].base_score == 9.3
    assert entries[0].base_severity == "CRITICAL"
    assert entries[0].source == "nvd"
    assert entries[0].is_authoritative is True


def test_parse_cvss_v40_score_zero() -> None:
    """CVSS 4.0 with score 0.0 computes NONE severity."""
    metrics = _make_nvd_metric(
        version="4.0", key_suffix="V40", base_score=0.0, base_severity=None
    )
    entries = parse_cvss(metrics)
    assert entries[0].base_severity == "NONE"
    assert entries[0].base_score == 0.0


def test_parse_cvss_v31_reads_severity_from_cvss_data() -> None:
    """CVSS 3.1: baseSeverity read from cvssData (not computed)."""
    metrics = _make_nvd_metric(version="3.1", key_suffix="V31", base_score=9.8)
    entries = parse_cvss(metrics)
    assert len(entries) == 1
    assert entries[0].version == "3.1"
    assert entries[0].base_score == 9.8
    assert entries[0].base_severity == "CRITICAL"


def test_parse_cvss_v2_severity_at_metric_level() -> None:
    """CVSS 2.0: baseSeverity at metric level, not in cvssData."""
    metrics = _make_nvd_metric(
        version="2.0",
        key_suffix="V2",
        base_score=4.3,
        base_severity="MEDIUM",
        severity_at_metric_level=True,
    )
    entries = parse_cvss(metrics)
    assert entries[0].base_severity == "MEDIUM"


def test_parse_cvss_multi_version_ordering() -> None:
    """Multiple CVSS versions parsed in order: v4.0, v3.1, v2.0."""
    metrics: dict[str, list[dict[str, Any]]] = {}
    metrics.update(
        _make_nvd_metric(version="4.0", key_suffix="V40", base_score=9.3, base_severity=None)
    )
    metrics.update(
        _make_nvd_metric(version="3.1", key_suffix="V31", base_score=9.8)
    )
    metrics.update(
        _make_nvd_metric(
            version="2.0", key_suffix="V2", base_score=7.5,
            base_severity="HIGH", severity_at_metric_level=True,
        )
    )
    entries = parse_cvss(metrics)
    assert len(entries) == 3
    assert entries[0].version == "4.0"
    assert entries[1].version == "3.1"
    assert entries[2].version == "2.0"


def test_parse_cvss_empty() -> None:
    """Empty metrics returns empty list."""
    assert parse_cvss({}) == []


# --- parse_cwes ---


def test_parse_cwes_basic() -> None:
    """Parse single CWE from weakness."""
    weaknesses = [{"description": [{"lang": "en", "value": "CWE-94"}]}]
    entries = parse_cwes(weaknesses)
    assert len(entries) == 1
    assert entries[0].id == "CWE-94"


def test_parse_cwes_skips_non_cwe() -> None:
    """Non-CWE values like NVD-CWE-noinfo are skipped."""
    weaknesses = [{"description": [{"lang": "en", "value": "NVD-CWE-noinfo"}]}]
    assert parse_cwes(weaknesses) == []


def test_parse_cwes_empty() -> None:
    """Empty weaknesses returns empty list."""
    assert parse_cwes([]) == []


# --- parse_cpes ---


def test_parse_cpes_basic() -> None:
    """Parse CPE with fix version from versionEndExcluding."""
    configurations = [
        {
            "nodes": [
                {
                    "cpeMatch": [
                        {
                            "criteria": "cpe:2.3:a:example:product:1.0:*:*:*:*:*:*:*",
                            "versionEndExcluding": "2.0.0",
                        }
                    ]
                }
            ]
        }
    ]
    entries = parse_cpes(configurations)
    assert len(entries) == 1
    assert entries[0].vendor == "example"
    assert entries[0].product == "product"
    assert entries[0].fix_version == "2.0.0"


def test_parse_cpes_skips_invalid_cpe_string() -> None:
    """Invalid CPE strings are silently skipped."""
    configurations = [
        {
            "nodes": [
                {
                    "cpeMatch": [
                        {"criteria": "cpe:2.3:INVALID"},
                        {"criteria": "cpe:2.3:a:valid:product:1.0:*:*:*:*:*:*:*"},
                    ]
                }
            ]
        }
    ]
    entries = parse_cpes(configurations)
    assert len(entries) == 1
    assert entries[0].vendor == "valid"


def test_parse_cpes_empty() -> None:
    """Empty configurations returns empty list."""
    assert parse_cpes([]) == []


# --- parse_references ---


def test_parse_references_basic() -> None:
    """Parse reference with tags."""
    refs = [
        {
            "url": "https://example.com/advisory",
            "source": "nvd@nist.gov",
            "tags": ["Vendor Advisory"],
        }
    ]
    entries = parse_references(refs)
    assert len(entries) == 1
    assert entries[0].url == "https://example.com/advisory"
    assert entries[0].tags == ["Vendor Advisory"]


def test_parse_references_no_tags() -> None:
    """Reference without tags gets empty list."""
    refs = [{"url": "https://example.com/advisory", "source": "nvd@nist.gov"}]
    entries = parse_references(refs)
    assert entries[0].tags == []


def test_parse_references_non_string_source_and_non_list_tags() -> None:
    """Non-string source becomes None, non-list tags becomes empty list."""
    refs = [
        {
            "url": "https://example.com/advisory",
            "source": 12345,
            "tags": "not-a-list",
        }
    ]
    entries = parse_references(refs)
    assert len(entries) == 1
    assert entries[0].source is None
    assert entries[0].tags == []


def test_parse_references_empty() -> None:
    """Empty references returns empty list."""
    assert parse_references([]) == []
