"""
Dataset Core - VulnDataset with columnar storage
Tests: Essential VulnDataset operations

Layer: Core ETL data structure
Dependencies: schema
"""

from __future__ import annotations

import numpy as np
import pytest

from vulnflow.dataset import VulnDataset


def test_dataset_from_nvd():
    """VulnDataset.from_nvd() loads NVD file."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")
    assert len(dataset) == 3
    assert "cve_id" in dataset.columns


def test_dataset_columns_property():
    """VulnDataset.columns returns column names."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")
    columns = dataset.columns

    assert isinstance(columns, list)
    assert "cve_id" in columns
    assert "description" in columns


def test_dataset_get_column():
    """VulnDataset.get_column() returns numpy array."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")
    cve_ids = dataset.get_column("cve_id")

    assert isinstance(cve_ids, np.ndarray)
    assert len(cve_ids) == 3


def test_dataset_get_column_missing_raises():
    """VulnDataset.get_column() raises KeyError for missing column."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    with pytest.raises(KeyError):
        dataset.get_column("nonexistent_column")


def test_dataset_set_column():
    """VulnDataset.set_column() adds new column."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    new_data = np.array(["NEW1", "NEW2", "NEW3"], dtype=object)
    dataset.set_column("computed_field", new_data)

    assert "computed_field" in dataset.columns
    assert list(dataset.get_column("computed_field")) == ["NEW1", "NEW2", "NEW3"]


def test_dataset_set_column_wrong_length_raises():
    """VulnDataset.set_column() raises ValueError for wrong length."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    wrong_data = np.array(["A", "B"], dtype=object)  # Should be 3

    with pytest.raises(ValueError, match="length mismatch"):
        dataset.set_column("bad_column", wrong_data)


def test_dataset_transform_applies_transforms():
    """VulnDataset.transform() applies transforms."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    class MockTransform:
        name = "mock"
        depends_on: list[str] = []

        def transform(self, ds: VulnDataset) -> None:
            ds.set_column("mock_column", np.array(["A", "B", "C"], dtype=object))

    dataset.transform([MockTransform()])

    assert "mock_column" in dataset.columns


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
        depends_on = ["t1"]

        def transform(self, dataset):
            execution_order.append("t2")

    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")
    dataset.transform([Transform2(), Transform1()])

    assert execution_order == ["t1", "t2"]


def test_dataset_transform_detects_circular_dependencies():
    """dataset.transform() raises on circular dependencies."""

    class Transform1:
        name = "t1"
        depends_on = ["t2"]

    class Transform2:
        name = "t2"
        depends_on = ["t1"]

    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    with pytest.raises(ValueError, match="Circular dependency"):
        dataset.transform([Transform1(), Transform2()])


def test_dataset_transform_runs_async_concurrently():
    """dataset.transform() runs async enrichers concurrently."""
    import asyncio
    import time

    class AsyncEnricher:
        name = "async1"
        depends_on: list[str] = []

        async def transform(self, dataset):
            await asyncio.sleep(0.1)

    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    start = time.time()
    dataset.transform([AsyncEnricher(), AsyncEnricher()])
    elapsed = time.time() - start

    # Should take ~0.1s (concurrent), not ~0.2s (sequential)
    assert elapsed < 0.15


def test_merge_multiple_datasets():
    """VulnDataset.merge() concatenates datasets."""
    ds1 = VulnDataset()
    ds1.set_column("cve_id", np.array(["CVE-1", "CVE-2"]))

    ds2 = VulnDataset()
    ds2.set_column("cve_id", np.array(["CVE-3"]))

    result = VulnDataset.merge([ds1, ds2])

    assert len(result) == 3
    assert list(result.get_column("cve_id")) == ["CVE-1", "CVE-2", "CVE-3"]


def test_merge_handles_missing_columns():
    """VulnDataset.merge() fills None for missing columns."""
    ds1 = VulnDataset()
    ds1.set_column("cve_id", np.array(["CVE-1"]))
    ds1.set_column("score", np.array([7.5]))

    ds2 = VulnDataset()
    ds2.set_column("cve_id", np.array(["CVE-2"]))

    result = VulnDataset.merge([ds1, ds2])

    assert len(result) == 2
    assert result.get_column("score")[0] == 7.5
    assert result.get_column("score")[1] is None
