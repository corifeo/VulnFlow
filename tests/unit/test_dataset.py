"""
Dataset Core - VulnDataset with columnar storage
Tests: VulnDataset creation, _from_records, __len__

Layer: Core ETL data structure
Dependencies: schema
"""

from __future__ import annotations

from vulnflow.dataset import VulnDataset
from vulnflow.schema import VulnRecord


def test_dataset_from_records():
    """VulnDataset._from_records converts list of VulnRecord to columnar."""
    records = [
        VulnRecord(
            cve_id="CVE-2024-0001",
            description="First vulnerability",
        ),
        VulnRecord(
            cve_id="CVE-2024-0002",
            description="Second vulnerability",
        ),
    ]
    dataset = VulnDataset._from_records(records)
    assert len(dataset) == 2
    assert dataset._data is not None
    assert "cve_id" in dataset._data


def test_dataset_len():
    """VulnDataset.__len__ returns record count."""
    records = [
        VulnRecord(
            cve_id="CVE-2024-0001",
            description="Test",
        )
    ]
    dataset = VulnDataset._from_records(records)
    assert len(dataset) == 1


def test_dataset_empty():
    """VulnDataset handles empty records."""
    dataset = VulnDataset._from_records([])
    assert len(dataset) == 0


def test_dataset_from_nvd():
    """VulnDataset.from_nvd() class method loads NVD file."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")
    assert len(dataset) == 3
    assert "cve_id" in dataset._data


def test_dataset_columnar_schema_complete():
    """VulnDataset._from_records stores all VulnRecord fields."""
    from datetime import UTC, datetime

    from vulnflow.schema import CVSSEntry

    cvss = CVSSEntry(
        version="3.1",
        vector_string="CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
        base_score=9.8,
        base_severity="CRITICAL",
        source="nvd",
    )
    records = [
        VulnRecord(
            cve_id="CVE-2024-0001",
            description="Test vulnerability",
            nvd_published_at=datetime(2024, 1, 1, tzinfo=UTC),
            cvss_scores=[cvss],
        )
    ]
    dataset = VulnDataset._from_records(records)

    # Core fields present
    expected_columns = [
        "id",
        "cve_id",
        "aliases",
        "title",
        "description",
        "first_seen_at",
        "last_updated_at",
        "sources",
        "nvd_published_at",
        "nvd_modified_at",
        "cvss_scores",
        "epss_scores",
        "kev",
        "exploits",
        "cwes",
        "cpes",
        "references",
        "tags",
        "metadata",
    ]
    for col in expected_columns:
        assert col in dataset._data, f"Missing column: {col}"

    # CVD timestamps should NOT be in dataset (added by EventInference later)
    cvd_columns = [
        "v_timestamp",
        "f_timestamp",
        "d_timestamp",
        "p_timestamp",
        "x_timestamp",
        "a_timestamp",
    ]
    for col in cvd_columns:
        assert col not in dataset._data, f"CVD column should not exist: {col}"

    # CVSS scores stored as JSON string
    import json

    cvss_json = dataset._data["cvss_scores"][0]
    cvss_data = json.loads(cvss_json)
    assert len(cvss_data) == 1
    assert cvss_data[0]["base_score"] == 9.8
    assert cvss_data[0]["source"] == "nvd"
