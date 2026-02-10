"""Tests for CVSS utilities."""

import pytest

from vulnflow.utils.cvss import (
    extract_cvss_metadata,
    normalize_source,
    score_to_severity,
)


def test_normalize_source_nvd():
    """Normalize NVD source variations."""
    assert normalize_source("nvd@nist.gov") == "nvd"
    assert normalize_source("NVD@nist.gov") == "nvd"


def test_normalize_source_vendors():
    """Normalize known vendor sources."""
    assert normalize_source("security@redhat.com") == "redhat"
    assert normalize_source("security@microsoft.com") == "microsoft"
    assert normalize_source("cve@mitre.org") == "mitre"


def test_normalize_source_unknown():
    """Extract domain from unknown email."""
    assert normalize_source("security@example.com") == "example"


def test_normalize_source_case_insensitive():
    """Test case insensitivity."""
    assert normalize_source("SECURITY@REDHAT.COM") == "redhat"
    assert normalize_source("Security@Microsoft.Com") == "microsoft"


def test_normalize_source_nist_variations():
    """Test nist.gov variations."""
    assert normalize_source("nvd@nist.gov") == "nvd"
    assert normalize_source("anything@nist.gov") == "nvd"


def test_normalize_source_no_at_sign():
    """Test source without @ sign."""
    assert normalize_source("nvd") == "nvd"
    assert normalize_source("redhat") == "redhat"
    assert normalize_source("unknown_source") == "unknown_source"


def test_extract_cvss_metadata_primary():
    """Extract metadata from Primary CVSS metric."""
    metric = {
        "source": "nvd@nist.gov",
        "type": "Primary",
        "exploitabilityScore": 3.9,
        "impactScore": 5.9,
    }

    result = extract_cvss_metadata(metric)

    assert result["source"] == "nvd"
    assert result["is_authoritative"] is True
    assert result["exploitability_score"] == 3.9
    assert result["impact_score"] == 5.9
    assert result["source_url"] is None
    assert result["published_at"] is None
    assert result["updated_at"] is None


def test_extract_cvss_metadata_secondary():
    """Extract metadata from Secondary CVSS metric."""
    metric = {
        "source": "security@redhat.com",
        "type": "Secondary",
    }

    result = extract_cvss_metadata(metric)

    assert result["source"] == "redhat"
    assert result["is_authoritative"] is False
    assert result["exploitability_score"] is None
    assert result["impact_score"] is None


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.0, "NONE"),
        (0.1, "LOW"),
        (3.9, "LOW"),
        (4.0, "MEDIUM"),
        (5.1, "MEDIUM"),
        (6.9, "MEDIUM"),
        (7.0, "HIGH"),
        (8.9, "HIGH"),
        (9.0, "CRITICAL"),
        (9.3, "CRITICAL"),
        (10.0, "CRITICAL"),
    ],
)
def test_score_to_severity(score: float, expected: str) -> None:
    """Map CVSS score to severity at all boundary values."""
    assert score_to_severity(score) == expected
