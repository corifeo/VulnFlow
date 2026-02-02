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


def test_dataset_columns_property():
    """VulnDataset.columns returns list of column names."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")
    columns = dataset.columns

    assert isinstance(columns, list)
    assert "cve_id" in columns
    assert "nvd_published_at" in columns
    assert "description" in columns


def test_dataset_get_column():
    """VulnDataset.get_column() returns numpy array."""
    import numpy as np

    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")
    cve_ids = dataset.get_column("cve_id")

    assert isinstance(cve_ids, np.ndarray)
    assert len(cve_ids) == 3
    assert cve_ids[0] == "CVE-2024-0001"
    assert cve_ids[1] == "CVE-2024-0002"


def test_dataset_get_column_missing_raises():
    """VulnDataset.get_column() raises KeyError for missing column."""
    import pytest

    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    with pytest.raises(KeyError):
        dataset.get_column("nonexistent_column")


def test_dataset_set_column():
    """VulnDataset.set_column() adds new column."""
    import numpy as np

    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    # Add computed column
    new_data = np.array(["NEW1", "NEW2", "NEW3"], dtype=object)
    dataset.set_column("computed_field", new_data)

    assert "computed_field" in dataset.columns
    assert list(dataset.get_column("computed_field")) == ["NEW1", "NEW2", "NEW3"]


def test_dataset_set_column_wrong_length_raises():
    """VulnDataset.set_column() raises ValueError for wrong length."""
    import numpy as np
    import pytest

    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    # Try to add column with wrong length
    wrong_data = np.array(["A", "B"], dtype=object)  # Should be 3

    with pytest.raises(ValueError, match="length mismatch"):
        dataset.set_column("bad_column", wrong_data)


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
