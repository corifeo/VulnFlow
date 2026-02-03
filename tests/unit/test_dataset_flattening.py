"""
Dataset Flattening Logic - Tests for CVSS/CWE/CPE flattening

Tests: Flattening logic for complex nested fields
Layer: Core ETL data structure
Dependencies: schema
"""

from __future__ import annotations

from tests.conftest import dataset_from_records

from vulnflow.schema import CPEEntry, CVSSEntry, CWEEntry, VulnRecord


def test_flatten_cvss_prefers_nvd_source():
    """Flatten CVSS prefers NVD over other sources."""
    records = [
        VulnRecord(
            cve_id="CVE-2024-0001",
            description="Test",
            cvss_scores=[
                CVSSEntry(
                    version="3.1",
                    vector_string="CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:L/I:L/A:L",
                    base_score=6.3,
                    base_severity="MEDIUM",
                    source="redhat",
                ),
                CVSSEntry(
                    version="3.1",
                    vector_string="CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
                    base_score=9.8,
                    base_severity="CRITICAL",
                    source="nvd",
                ),
            ],
        )
    ]
    dataset = dataset_from_records(records)

    # Should prefer NVD source
    assert dataset.get_column("cvss_base_score")[0] == 9.8
    assert dataset.get_column("cvss_base_severity")[0] == "CRITICAL"
    assert dataset.get_column("cvss_source")[0] == "nvd"


def test_flatten_cvss_falls_back_to_redhat():
    """Flatten CVSS falls back to RedHat when NVD unavailable."""
    records = [
        VulnRecord(
            cve_id="CVE-2024-0001",
            description="Test",
            cvss_scores=[
                CVSSEntry(
                    version="3.1",
                    vector_string="CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:L/I:L/A:L",
                    base_score=5.0,
                    base_severity="MEDIUM",
                    source="microsoft",
                ),
                CVSSEntry(
                    version="3.1",
                    vector_string="CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:L/A:L",
                    base_score=7.3,
                    base_severity="HIGH",
                    source="redhat",
                ),
            ],
        )
    ]
    dataset = dataset_from_records(records)

    # Should prefer RedHat over Microsoft
    assert dataset.get_column("cvss_base_score")[0] == 7.3
    assert dataset.get_column("cvss_base_severity")[0] == "HIGH"
    assert dataset.get_column("cvss_source")[0] == "redhat"


def test_flatten_cvss_falls_back_to_first():
    """Flatten CVSS takes first entry when no preferred source."""
    records = [
        VulnRecord(
            cve_id="CVE-2024-0001",
            description="Test",
            cvss_scores=[
                CVSSEntry(
                    version="3.1",
                    vector_string="CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:L/I:L/A:L",
                    base_score=6.5,
                    base_severity="MEDIUM",
                    source="vendor",
                ),
                CVSSEntry(
                    version="3.1",
                    vector_string="CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:L/A:L",
                    base_score=7.3,
                    base_severity="HIGH",
                    source="other",
                ),
            ],
        )
    ]
    dataset = dataset_from_records(records)

    # Should take first when no preferred source
    assert dataset.get_column("cvss_base_score")[0] == 6.5
    assert dataset.get_column("cvss_base_severity")[0] == "MEDIUM"
    assert dataset.get_column("cvss_source")[0] == "vendor"


def test_flatten_cwe_takes_first_entry():
    """Flatten CWE takes first entry from list."""
    records = [
        VulnRecord(
            cve_id="CVE-2024-0001",
            description="Test",
            cwes=[
                CWEEntry(id="CWE-79", name="Cross-site Scripting"),
                CWEEntry(id="CWE-89", name="SQL Injection"),
            ],
        )
    ]
    dataset = dataset_from_records(records)

    assert dataset.get_column("primary_cwe_id")[0] == "CWE-79"
    assert dataset.get_column("primary_cwe_name")[0] == "Cross-site Scripting"


def test_flatten_cpe_takes_first_entry():
    """Flatten CPE takes first entry from list."""
    records = [
        VulnRecord(
            cve_id="CVE-2024-0001",
            description="Test",
            cpes=[
                CPEEntry(vendor="apache", product="log4j", version="2.14.1"),
                CPEEntry(vendor="apache", product="struts", version="2.5.20"),
            ],
        )
    ]
    dataset = dataset_from_records(records)

    assert dataset.get_column("primary_vendor")[0] == "apache"
    assert dataset.get_column("primary_product")[0] == "log4j"
    assert dataset.get_column("primary_version")[0] == "2.14.1"
