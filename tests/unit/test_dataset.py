"""
Dataset Core - VulnDataset with columnar storage
Tests: VulnDataset creation, __len__, columns

Layer: Core ETL data structure
Dependencies: schema
"""

from __future__ import annotations

import pytest
from tests.conftest import dataset_from_records

from vulnflow.dataset import VulnDataset
from vulnflow.schema import VulnRecord


def test_dataset_from_records():
    """Test helper dataset_from_records converts list of VulnRecord to columnar."""
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
    dataset = dataset_from_records(records)
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
    dataset = dataset_from_records(records)
    assert len(dataset) == 1


def test_dataset_empty():
    """VulnDataset handles empty records."""
    dataset = dataset_from_records([])
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

    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    # Try to add column with wrong length
    wrong_data = np.array(["A", "B"], dtype=object)  # Should be 3

    with pytest.raises(ValueError, match="length mismatch"):
        dataset.set_column("bad_column", wrong_data)


def test_dataset_transform_applies_transforms():
    """VulnDataset.transform() applies transforms."""
    import numpy as np

    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    # Mock transform that adds a column
    class MockTransform:
        name = "mock"
        depends_on: list[str] = []

        def transform(self, ds: VulnDataset) -> None:
            ds.set_column("mock_column", np.array(["A", "B", "C"], dtype=object))

    dataset.transform([MockTransform()])

    assert "mock_column" in dataset.columns
    assert list(dataset.get_column("mock_column")) == ["A", "B", "C"]


def test_dataset_transform_empty_list():
    """VulnDataset.transform() with empty list does nothing."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")
    initial_columns = dataset.columns.copy()

    dataset.transform([])

    assert dataset.columns == initial_columns


def test_dataset_columnar_schema_complete():
    """dataset_from_records stores all VulnRecord fields."""
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
    dataset = dataset_from_records(records)

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


def test_dataset_transform_sorts_by_dependencies():
    """dataset.transform() sorts transforms by depends_on."""

    execution_order = []

    class Transform1:
        name = "t1"
        depends_on: list[str] = []

        def transform(self, dataset):
            execution_order.append("t1")

    class Transform2:
        name = "t2"
        depends_on = ["t1"]  # Depends on t1

        def transform(self, dataset):
            execution_order.append("t2")

    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    # Pass in wrong order - should auto-sort
    dataset.transform([Transform2(), Transform1()])

    # Should execute in dependency order
    assert execution_order == ["t1", "t2"]


def test_dataset_transform_detects_circular_dependencies():
    """dataset.transform() raises on circular dependencies."""

    class Transform1:
        name = "t1"
        depends_on = ["t2"]

    class Transform2:
        name = "t2"
        depends_on = ["t1"]  # Circular!

    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    with pytest.raises(ValueError, match="Circular dependency"):
        dataset.transform([Transform1(), Transform2()])


def test_dataset_transform_runs_async_concurrently():
    """dataset.transform() runs async enrichers concurrently."""
    import asyncio
    import time

    call_log = []

    class AsyncEnricher1:
        name = "async1"
        depends_on: list[str] = []

        async def transform(self, dataset):
            call_log.append(("async1_start", time.time()))
            await asyncio.sleep(0.1)
            call_log.append(("async1_end", time.time()))

    class AsyncEnricher2:
        name = "async2"
        depends_on: list[str] = []

        async def transform(self, dataset):
            call_log.append(("async2_start", time.time()))
            await asyncio.sleep(0.1)
            call_log.append(("async2_end", time.time()))

    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    start = time.time()
    dataset.transform([AsyncEnricher1(), AsyncEnricher2()])
    elapsed = time.time() - start

    # Should take ~0.1s (concurrent), not ~0.2s (sequential)
    assert elapsed < 0.15, f"Async enrichers ran sequentially ({elapsed:.2f}s)"

    # Both should start before either ends (overlapping execution)
    assert call_log[0][0] in ["async1_start", "async2_start"]
    assert call_log[1][0] in ["async1_start", "async2_start"]


def test_from_records_handles_empty_cvss_list():
    """dataset_from_records handles VulnRecord with empty cvss_scores list."""
    records = [
        VulnRecord(
            cve_id="CVE-2024-0001",
            description="Test",
            cvss_scores=[],  # Empty list
        )
    ]
    dataset = dataset_from_records(records)

    assert dataset.get_column("cvss_base_score")[0] is None
    assert dataset.get_column("cvss_base_severity")[0] is None
    assert dataset.get_column("cvss_version")[0] is None
    assert dataset.get_column("cvss_vector_string")[0] is None
    assert dataset.get_column("cvss_source")[0] is None


def test_from_records_handles_empty_cwe_list():
    """dataset_from_records handles VulnRecord with empty cwes list."""
    records = [
        VulnRecord(
            cve_id="CVE-2024-0001",
            description="Test",
            cwes=[],  # Empty list
        )
    ]
    dataset = dataset_from_records(records)

    assert dataset.get_column("primary_cwe_id")[0] is None
    assert dataset.get_column("primary_cwe_name")[0] is None


def test_from_records_handles_empty_cpe_list():
    """dataset_from_records handles VulnRecord with empty cpes list."""
    records = [
        VulnRecord(
            cve_id="CVE-2024-0001",
            description="Test",
            cpes=[],  # Empty list
        )
    ]
    dataset = dataset_from_records(records)

    assert dataset.get_column("primary_vendor")[0] is None
    assert dataset.get_column("primary_product")[0] is None
    assert dataset.get_column("primary_version")[0] is None


def test_from_records_handles_none_values():
    """dataset_from_records handles VulnRecord with None in optional fields."""
    records = [
        VulnRecord(
            cve_id="CVE-2024-0001",
            description="Test",
            title=None,
            nvd_published_at=None,
            kev=None,
        )
    ]
    dataset = dataset_from_records(records)

    assert dataset.get_column("title")[0] is None
    assert dataset.get_column("nvd_published_at")[0] is None
    assert dataset.get_column("kev")[0] == "null"  # JSON-encoded None


def test_from_records_json_encoding_handles_empty_lists():
    """dataset_from_records JSON-encodes empty lists correctly."""
    import json

    records = [
        VulnRecord(
            cve_id="CVE-2024-0001",
            description="Test",
            aliases=[],
            sources=[],
            tags={},
        )
    ]
    dataset = dataset_from_records(records)

    assert json.loads(dataset.get_column("aliases")[0]) == []
    assert json.loads(dataset.get_column("sources")[0]) == []
    assert json.loads(dataset.get_column("tags")[0]) == {}


def test_transform_returns_self_for_chaining():
    """transform() returns self to enable method chaining."""
    import numpy as np

    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    class MockTransform:
        name = "mock"
        depends_on: list[str] = []

        def transform(self, ds: VulnDataset) -> None:
            ds.set_column("mock_column", np.array(["A", "B", "C"], dtype=object))

    result = dataset.transform([MockTransform()])

    # Should return self for chaining
    assert result is dataset


def test_method_chaining_transform_to_pandas():
    """Method chaining works: transform().to_pandas()."""
    import numpy as np

    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    class MockTransform:
        name = "mock"
        depends_on: list[str] = []

        def transform(self, ds: VulnDataset) -> None:
            ds.set_column("mock_column", np.array(["A", "B", "C"], dtype=object))

    # Should be able to chain
    df = dataset.transform([MockTransform()]).to_pandas()

    assert "mock_column" in df.columns
    assert list(df["mock_column"]) == ["A", "B", "C"]


def test_to_pandas_delegates_to_pandas_loader():
    """dataset.to_pandas() delegates to PandasLoader."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    df = dataset.to_pandas()

    # Should have flattened columns (proof of PandasLoader usage)
    assert "cvss_base_score" in df.columns
    assert "primary_cwe_id" in df.columns
    assert "primary_vendor" in df.columns

    # Should NOT include nested JSON columns (PandasLoader default behavior)
    assert "references" not in df.columns
    assert "exploits" not in df.columns
    assert "epss_scores" not in df.columns
    assert "kev" not in df.columns


def test_to_polars_delegates_to_polars_loader():
    """dataset.to_polars() delegates to PolarsLoader."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    df = dataset.to_polars()

    # Should have flattened columns (proof of PolarsLoader usage)
    assert "cvss_base_score" in df.columns
    assert "primary_cwe_id" in df.columns
    assert "primary_vendor" in df.columns

    # Should NOT include nested JSON columns (PolarsLoader default behavior)
    assert "references" not in df.columns
    assert "exploits" not in df.columns
    assert "epss_scores" not in df.columns
    assert "kev" not in df.columns
