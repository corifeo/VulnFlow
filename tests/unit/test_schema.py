"""
Schema - VulnRecord and nested dataclasses
Tests: VulnRecord, CVSSEntry, EPSSEntry, KEVEntry creation

Layer: Core data schema
Dependencies: None
"""

from __future__ import annotations

from datetime import UTC, datetime

from vulnflow.schema import CVSSEntry, VulnRecord


def test_vuln_record_minimal():
    """VulnRecord can be created with minimal required fields."""
    record = VulnRecord(
        cve_id="CVE-2024-1234",
        description="Test vulnerability",
    )
    assert record.cve_id == "CVE-2024-1234"
    assert record.description == "Test vulnerability"
    assert record.nvd_published_at is None
    assert record.nvd_modified_at is None
    assert record.cvss_scores == []


def test_vuln_record_with_cvss():
    """VulnRecord accepts nested CVSSEntry list."""
    cvss = CVSSEntry(
        version="3.1",
        vector_string="CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
        base_score=9.8,
        base_severity="CRITICAL",
        source="nvd",
    )
    record = VulnRecord(
        cve_id="CVE-2024-1234",
        description="Test vulnerability",
        cvss_scores=[cvss],
    )
    assert len(record.cvss_scores) == 1
    assert record.cvss_scores[0].base_score == 9.8
    assert record.cvss_scores[0].base_severity == "CRITICAL"


def test_vuln_record_with_nvd_timestamps():
    """VulnRecord accepts NVD temporal fields."""
    record = VulnRecord(
        cve_id="CVE-2024-1234",
        description="Test vulnerability",
        nvd_published_at=datetime(2024, 1, 1, tzinfo=UTC),
        nvd_modified_at=datetime(2024, 1, 2, tzinfo=UTC),
    )
    assert record.nvd_published_at is not None
    assert record.nvd_published_at.year == 2024
    assert record.nvd_modified_at is not None
