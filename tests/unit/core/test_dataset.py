"""
Dataset Core - VulnDataset with columnar storage
Tests: Essential VulnDataset operations

Layer: Core ETL data structure
Dependencies: schema
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vulnflow.dataset import VulnDataset


def test_dataset_from_nvd():
    """VulnDataset.from_nvd() loads NVD file."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")
    assert len(dataset) == 5
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
    assert len(cve_ids) == 5


def test_dataset_get_column_missing_raises():
    """VulnDataset.get_column() raises KeyError for missing column."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    with pytest.raises(KeyError):
        dataset.get_column("nonexistent_column")


def test_dataset_set_column():
    """VulnDataset.set_column() adds new column."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    new_data = np.array(["NEW1", "NEW2", "NEW3", "NEW4", "NEW5"], dtype=object)
    dataset.set_column("computed_field", new_data)

    assert "computed_field" in dataset.columns
    assert list(dataset.get_column("computed_field")) == [
        "NEW1",
        "NEW2",
        "NEW3",
        "NEW4",
        "NEW5",
    ]


def test_dataset_set_column_wrong_length_raises():
    """VulnDataset.set_column() raises ValueError for wrong length."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    wrong_data = np.array(["A", "B"], dtype=object)  # Should be 5

    with pytest.raises(ValueError, match="length mismatch"):
        dataset.set_column("bad_column", wrong_data)


def test_dataset_transform_applies_transforms():
    """VulnDataset.transform() applies transforms."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    class MockTransform:
        name = "mock"
        depends_on: list[str] = []

        def transform(self, ds: VulnDataset) -> None:
            ds.set_column("mock_column", np.array(["A"] * len(ds), dtype=object))

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


def test_merge_preserves_column_order():
    """VulnDataset.merge() preserves first dataset's column order, appending new columns."""
    # First dataset has columns in order: id, cve_id, score
    ds1 = VulnDataset()
    ds1.set_column("id", np.array(["1", "2"]))
    ds1.set_column("cve_id", np.array(["CVE-1", "CVE-2"]))
    ds1.set_column("score", np.array([7.5, 8.0]))

    # Second dataset has different column order: severity, cve_id, tags
    ds2 = VulnDataset()
    ds2.set_column("severity", np.array(["HIGH"]))
    ds2.set_column("cve_id", np.array(["CVE-3"]))
    ds2.set_column("tags", np.array(["test"]))

    result = VulnDataset.merge([ds1, ds2])

    # Should preserve ds1's order (id, cve_id, score) and append new columns (severity, tags)
    expected_order = ["id", "cve_id", "score", "severity", "tags"]
    assert result.columns == expected_order


def test_merge_repeated_is_stable():
    """VulnDataset.merge() produces identical column order when repeated."""
    ds1 = VulnDataset()
    ds1.set_column("a", np.array([1]))
    ds1.set_column("b", np.array([2]))

    ds2 = VulnDataset()
    ds2.set_column("c", np.array([3]))
    ds2.set_column("d", np.array([4]))

    # Merge multiple times
    result1 = VulnDataset.merge([ds1, ds2])
    result2 = VulnDataset.merge([ds1, ds2])
    result3 = VulnDataset.merge([ds1, ds2])

    # All should have identical column order
    assert result1.columns == result2.columns
    assert result2.columns == result3.columns
    assert result1.columns == ["a", "b", "c", "d"]


class TestProvenance:
    """Tests for VulnDataset provenance tracking."""

    def test_empty_dataset_has_empty_provenance(self) -> None:
        ds = VulnDataset()
        prov = ds.provenance
        assert prov["extractors"] == []
        assert prov["transforms"] == []
        assert "created_at" in prov
        assert "updated_at" in prov
        assert prov["record_count"] == 0
        assert prov["vulnflow_version"] == "0.1.0"

    def test_provenance_returns_deep_copy(self) -> None:
        ds = VulnDataset()
        prov1 = ds.provenance
        prov1["extractors"].append("fake")
        prov2 = ds.provenance
        assert prov2["extractors"] == []

    def test_record_extractor(self) -> None:
        ds = VulnDataset()
        ds._record_provenance(extractor="NVDExtractor(test.json)")
        assert ds.provenance["extractors"] == ["NVDExtractor(test.json)"]

    def test_record_transform(self) -> None:
        ds = VulnDataset()
        ds._record_provenance(transform="KEVEnricher")
        assert ds.provenance["transforms"] == ["KEVEnricher"]

    def test_record_updates_timestamp(self) -> None:
        ds = VulnDataset()
        t1 = ds.provenance["updated_at"]
        ds._record_provenance(transform="KEVEnricher")
        t2 = ds.provenance["updated_at"]
        assert t2 >= t1

    def test_record_deduplicates(self) -> None:
        ds = VulnDataset()
        ds._record_provenance(extractor="NVDExtractor(test.json)")
        ds._record_provenance(extractor="NVDExtractor(test.json)")
        assert len(ds.provenance["extractors"]) == 1

    def test_transform_records_provenance(
        self, nvd_sample_dataset: VulnDataset
    ) -> None:
        from vulnflow.transforms.computed.cvd_state import CVDStateTransform

        nvd_sample_dataset.transform([CVDStateTransform()])
        assert "CVDStateTransform" in nvd_sample_dataset.provenance["transforms"]

    def test_multiple_transforms_all_recorded(
        self, nvd_sample_dataset: VulnDataset
    ) -> None:
        from vulnflow.transforms.computed.cvd_state import CVDStateTransform
        from vulnflow.transforms.flatten.cvss import CVSSFlattenTransform

        nvd_sample_dataset.transform([CVDStateTransform()])
        nvd_sample_dataset.transform([CVSSFlattenTransform()])
        transforms = nvd_sample_dataset.provenance["transforms"]
        assert "CVDStateTransform" in transforms
        assert "CVSSFlattenTransform" in transforms

    def test_duplicate_transforms_not_repeated(
        self, nvd_sample_dataset: VulnDataset
    ) -> None:
        from vulnflow.transforms.computed.cvd_state import CVDStateTransform

        nvd_sample_dataset.transform([CVDStateTransform()])
        nvd_sample_dataset.transform([CVDStateTransform()])
        transforms = nvd_sample_dataset.provenance["transforms"]
        assert transforms.count("CVDStateTransform") == 1

    def test_merge_combines_extractors(self) -> None:
        ds1 = VulnDataset()
        ds1._data = {"cve_id": np.array(["CVE-2024-0001"])}
        ds1._record_provenance(extractor="NVDExtractor(2024.json.gz)")

        ds2 = VulnDataset()
        ds2._data = {"cve_id": np.array(["CVE-2025-0001"])}
        ds2._record_provenance(extractor="NVDExtractor(2025.json.gz)")

        merged = VulnDataset.merge([ds1, ds2])
        extractors = merged.provenance["extractors"]
        assert "NVDExtractor(2024.json.gz)" in extractors
        assert "NVDExtractor(2025.json.gz)" in extractors

    def test_merge_combines_transforms(self) -> None:
        ds1 = VulnDataset()
        ds1._data = {"cve_id": np.array(["CVE-2024-0001"])}
        ds1._record_provenance(transform="KEVEnricher")

        ds2 = VulnDataset()
        ds2._data = {"cve_id": np.array(["CVE-2025-0001"])}
        ds2._record_provenance(transform="EPSSEnricher")

        merged = VulnDataset.merge([ds1, ds2])
        transforms = merged.provenance["transforms"]
        assert "KEVEnricher" in transforms
        assert "EPSSEnricher" in transforms

    def test_merge_deduplicates_extractors(self) -> None:
        ds1 = VulnDataset()
        ds1._data = {"cve_id": np.array(["CVE-2024-0001"])}
        ds1._record_provenance(extractor="NVDExtractor(2024.json.gz)")

        ds2 = VulnDataset()
        ds2._data = {"cve_id": np.array(["CVE-2025-0001"])}
        ds2._record_provenance(extractor="NVDExtractor(2024.json.gz)")

        merged = VulnDataset.merge([ds1, ds2])
        assert merged.provenance["extractors"].count("NVDExtractor(2024.json.gz)") == 1

    def test_merge_uses_earliest_created_at(self) -> None:
        ds1 = VulnDataset()
        ds1._data = {"cve_id": np.array(["CVE-2024-0001"])}
        ds1._provenance["created_at"] = "2026-01-01T00:00:00+00:00"

        ds2 = VulnDataset()
        ds2._data = {"cve_id": np.array(["CVE-2025-0001"])}
        ds2._provenance["created_at"] = "2026-02-01T00:00:00+00:00"

        merged = VulnDataset.merge([ds1, ds2])
        assert merged.provenance["created_at"] == "2026-01-01T00:00:00+00:00"

    def test_merge_single_dataset_preserves_provenance(self) -> None:
        ds = VulnDataset()
        ds._data = {"cve_id": np.array(["CVE-2024-0001"])}
        ds._record_provenance(extractor="NVDExtractor(test.json)")

        merged = VulnDataset.merge([ds])
        assert merged.provenance["extractors"] == ["NVDExtractor(test.json)"]

    def test_parquet_round_trip_preserves_provenance(self, tmp_path: Path) -> None:
        """Provenance survives to_parquet -> from_parquet."""
        ds = VulnDataset()
        ds._data = {"cve_id": np.array(["CVE-2024-0001"])}
        ds._record_provenance(extractor="NVDExtractor(test.json)")
        ds._record_provenance(transform="KEVEnricher")

        path = tmp_path / "test.parquet"
        ds.to_parquet(path)

        loaded = VulnDataset.from_parquet(str(path))
        prov = loaded.provenance
        assert "NVDExtractor(test.json)" in prov["extractors"]
        assert "KEVEnricher" in prov["transforms"]
        assert prov["vulnflow_version"] == "0.1.0"

    def test_parquet_without_provenance_gets_defaults(self, tmp_path: Path) -> None:
        """Loading old parquet files without provenance metadata still works."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        # Write a bare parquet file without provenance metadata
        table = pa.table({"cve_id": ["CVE-2024-0001"]})
        pq.write_table(table, tmp_path / "legacy.parquet")

        loaded = VulnDataset.from_parquet(str(tmp_path / "legacy.parquet"))
        # Should have default provenance (with ParquetExtractor recorded)
        assert loaded.provenance["vulnflow_version"] == "0.1.0"
        assert "ParquetExtractor(" in loaded.provenance["extractors"][0]

    def test_parquet_round_trip_adds_parquet_extractor(self, tmp_path: Path) -> None:
        """Loading from parquet adds ParquetExtractor to provenance."""
        ds = VulnDataset()
        ds._data = {"cve_id": np.array(["CVE-2024-0001"])}
        ds._record_provenance(extractor="NVDExtractor(test.json)")

        path = tmp_path / "test.parquet"
        ds.to_parquet(path)

        loaded = VulnDataset.from_parquet(str(path))
        extractors = loaded.provenance["extractors"]
        assert "NVDExtractor(test.json)" in extractors
        assert any("ParquetExtractor(" in e for e in extractors)
