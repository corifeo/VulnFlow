"""
Dataset Core - VulnDataset with columnar storage
Tests: Essential VulnDataset operations

Layer: Core ETL data structure
Dependencies: schema
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import vulnflow
from vulnflow.dataset import VulnDataset
from vulnflow.exceptions import ExtractorError, LoaderError


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


def test_dataset_transform_quiet_suppresses_output(
    capsys: pytest.CaptureFixture[str],
):
    """VulnDataset.transform(quiet=True) produces no progress output."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    class MockTransform:
        name = "mock_quiet"
        depends_on: list[str] = []

        def transform(self, ds: VulnDataset) -> None:
            ds.set_column("quiet_col", np.array(["Q"] * len(ds), dtype=object))

    dataset.transform([MockTransform()], quiet=True)

    captured = capsys.readouterr()
    assert "+" not in captured.out
    assert "mock_quiet" not in captured.out
    assert "quiet_col" in dataset.columns


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


def test_merge_duplicate_cve_keeps_newest_modified():
    """VulnDataset.merge() keeps one row per cve_id: the newest nvd_modified_at wins."""
    cache = VulnDataset()
    cache.set_column("cve_id", np.array(["CVE-1", "CVE-2"], dtype=object))
    cache.set_column(
        "nvd_modified_at",
        np.array(["2026-11-01", "2026-11-01"], dtype="datetime64[us]"),
    )
    cache.set_column("score", np.array([5.0, 6.0]))

    delta = VulnDataset()
    delta.set_column("cve_id", np.array(["CVE-2", "CVE-3"], dtype=object))
    delta.set_column(
        "nvd_modified_at",
        np.array(["2026-11-09", "2026-11-09"], dtype="datetime64[us]"),
    )
    delta.set_column("score", np.array([9.0, 7.0]))

    for datasets in ([cache, delta], [delta, cache]):  # order must not matter
        result = VulnDataset.merge(datasets)
        scores = dict(
            zip(result.get_column("cve_id"), result.get_column("score"), strict=True)
        )
        assert len(result) == 3
        assert scores == {"CVE-1": 5.0, "CVE-2": 9.0, "CVE-3": 7.0}


def test_merge_duplicate_cve_without_dates_later_wins():
    """VulnDataset.merge() without nvd_modified_at keeps the later dataset's row."""
    ds1 = VulnDataset()
    ds1.set_column("cve_id", np.array(["CVE-1"], dtype=object))
    ds1.set_column("title", np.array(["old"], dtype=object))
    ds2 = VulnDataset()
    ds2.set_column("cve_id", np.array(["CVE-1"], dtype=object))
    ds2.set_column("title", np.array(["new"], dtype=object))

    result = VulnDataset.merge([ds1, ds2])

    assert result.get_column("title").tolist() == ["new"]


def test_insert_columns_after_replaces_existing_column():
    """Re-running a flatten transform must overwrite its earlier output, keeping position."""
    ds = VulnDataset()
    ds.set_column("cve_id", np.array(["CVE-1"], dtype=object))
    ds.set_column("epss_scores", np.array(["[]"], dtype=object))
    ds.set_column("epss_date", np.array(["2026-11-06"], dtype=object))
    ds.set_column("cwes", np.array(["[]"], dtype=object))

    ds.insert_columns_after(
        "epss_scores", {"epss_date": np.array(["2026-11-13"], dtype=object)}
    )

    assert ds.get_column("epss_date").tolist() == ["2026-11-13"]
    assert ds.columns == ["cve_id", "epss_scores", "epss_date", "cwes"]


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
        assert prov["vulnflow_version"] == vulnflow.__version__

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
        from vulnflow.transforms.inferences import EventInference

        nvd_sample_dataset.transform([EventInference(), CVDStateTransform()])
        assert "CVDStateTransform()" in nvd_sample_dataset.provenance["transforms"]

    def test_multiple_transforms_all_recorded(
        self, nvd_sample_dataset: VulnDataset
    ) -> None:
        from vulnflow.transforms.computed.cvd_state import CVDStateTransform
        from vulnflow.transforms.flatten.cvss import CVSSFlattenTransform
        from vulnflow.transforms.inferences import EventInference

        nvd_sample_dataset.transform([EventInference(), CVDStateTransform()])
        nvd_sample_dataset.transform([CVSSFlattenTransform()])
        transforms = nvd_sample_dataset.provenance["transforms"]
        assert "CVDStateTransform()" in transforms
        assert "CVSSFlattenTransform()" in transforms

    def test_duplicate_transforms_not_repeated(
        self, nvd_sample_dataset: VulnDataset
    ) -> None:
        from vulnflow.transforms.computed.cvd_state import CVDStateTransform
        from vulnflow.transforms.inferences import EventInference

        nvd_sample_dataset.transform([EventInference(), CVDStateTransform()])
        nvd_sample_dataset.transform([CVDStateTransform()])
        transforms = nvd_sample_dataset.provenance["transforms"]
        assert transforms.count("CVDStateTransform()") == 1

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
        assert prov["vulnflow_version"] == vulnflow.__version__

    def test_parquet_without_provenance_gets_defaults(self, tmp_path: Path) -> None:
        """Loading old parquet files without provenance metadata still works."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        # Write a bare parquet file without provenance metadata
        table = pa.table({"cve_id": ["CVE-2024-0001"]})
        pq.write_table(table, tmp_path / "legacy.parquet")

        loaded = VulnDataset.from_parquet(str(tmp_path / "legacy.parquet"))
        # Should have default provenance (with ParquetExtractor recorded)
        assert loaded.provenance["vulnflow_version"] == vulnflow.__version__
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


class TestFilter:
    """Tests for VulnDataset.filter() boolean mask selection."""

    def _make_dataset(self) -> VulnDataset:
        """Create a small dataset for filter tests."""
        ds = VulnDataset()
        ds.set_column("cve_id", np.array(["CVE-1", "CVE-2", "CVE-3"], dtype=object))
        ds.set_column("score", np.array([3.0, 7.5, 9.0]))
        ds._record_provenance(extractor="test")
        ds._record_provenance(transform="mock_transform")
        return ds

    def test_filter_returns_new_dataset(self) -> None:
        """filter() returns a new VulnDataset, not self."""
        ds = self._make_dataset()
        mask = np.array([True, True, True])
        result = ds.filter(mask)
        assert result is not ds

    def test_filter_selects_matching_rows(self) -> None:
        """filter() selects only rows where mask is True."""
        ds = self._make_dataset()
        mask = np.array([False, True, True])
        result = ds.filter(mask)
        assert len(result) == 2
        assert list(result.get_column("cve_id")) == ["CVE-2", "CVE-3"]
        assert list(result.get_column("score")) == [7.5, 9.0]

    def test_filter_all_true_returns_full_copy(self) -> None:
        """filter(all-True) preserves all rows."""
        ds = self._make_dataset()
        mask = np.ones(len(ds), dtype=bool)
        result = ds.filter(mask)
        assert len(result) == 3
        assert list(result.get_column("cve_id")) == ["CVE-1", "CVE-2", "CVE-3"]

    def test_filter_all_false_returns_empty(self) -> None:
        """filter(all-False) returns 0-row dataset with columns preserved."""
        ds = self._make_dataset()
        mask = np.zeros(len(ds), dtype=bool)
        result = ds.filter(mask)
        assert len(result) == 0
        assert result.columns == ["cve_id", "score"]

    def test_filter_preserves_all_columns(self) -> None:
        """filter() preserves column names and order."""
        ds = self._make_dataset()
        mask = np.array([True, False, True])
        result = ds.filter(mask)
        assert result.columns == ds.columns

    def test_filter_copies_provenance(self) -> None:
        """filter() carries over provenance from source dataset."""
        ds = self._make_dataset()
        mask = np.array([True, True, False])
        result = ds.filter(mask)
        assert "test" in result.provenance["extractors"]
        assert "mock_transform" in result.provenance["transforms"]

    def test_filter_provenance_is_deep_copy(self) -> None:
        """filter() provenance is independent from source."""
        ds = self._make_dataset()
        mask = np.array([True, True, True])
        result = ds.filter(mask)
        # Mutating result's internal provenance should not affect source
        result._provenance["extractors"].append("injected")
        assert "injected" not in ds.provenance["extractors"]

    def test_filter_wrong_type_raises_typeerror(self) -> None:
        """filter() rejects non-ndarray input."""
        ds = self._make_dataset()
        with pytest.raises(TypeError):
            ds.filter([True, True, True])  # type: ignore[arg-type]

    def test_filter_wrong_dtype_raises_valueerror(self) -> None:
        """filter() rejects non-boolean array."""
        ds = self._make_dataset()
        with pytest.raises(ValueError, match="boolean"):
            ds.filter(np.array([1, 0, 1]))

    def test_filter_wrong_length_raises_valueerror(self) -> None:
        """filter() rejects mask with wrong length."""
        ds = self._make_dataset()
        with pytest.raises(ValueError, match="length"):
            ds.filter(np.array([True, False]))


def _utc(year: int, month: int, day: int) -> datetime:
    """Shorthand for UTC-aware datetime."""
    return datetime(year, month, day, tzinfo=UTC)


class TestCountBy:
    """Tests for VulnDataset.count_by() period aggregation."""

    def _make_dataset(
        self,
        dates: list[datetime | None],
        statuses: list[str] | None = None,
    ) -> VulnDataset:
        """Create a dataset with given dates and optional statuses."""
        ds = VulnDataset()
        n = len(dates)
        ds.set_column(
            "cve_id",
            np.array([f"CVE-2024-{i:04d}" for i in range(n)], dtype=object),
        )
        ds.set_column("nvd_published_at", np.array(dates, dtype=object))
        if statuses is not None:
            ds.set_column("nvd_status", np.array(statuses, dtype=object))
        return ds

    def test_count_by_month_returns_dataframe(self) -> None:
        """count_by('month') returns a pandas DataFrame with correct columns."""
        ds = self._make_dataset([_utc(2024, 1, 15), _utc(2024, 2, 20)])
        result = ds.count_by("month")
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["month", "count"]

    def test_count_by_month_groups_correctly(self) -> None:
        """count_by('month') groups and counts correctly."""
        ds = self._make_dataset(
            [
                _utc(2024, 1, 10),
                _utc(2024, 1, 20),
                _utc(2024, 2, 5),
            ]
        )
        result = ds.count_by("month")
        assert len(result) == 2
        assert result.iloc[0]["month"] == "2024-01"
        assert result.iloc[0]["count"] == 2
        assert result.iloc[1]["month"] == "2024-02"
        assert result.iloc[1]["count"] == 1

    def test_count_by_year(self) -> None:
        """count_by('year') groups by year."""
        ds = self._make_dataset(
            [
                _utc(2023, 6, 1),
                _utc(2024, 1, 1),
                _utc(2024, 6, 1),
            ]
        )
        result = ds.count_by("year")
        assert list(result.columns) == ["year", "count"]
        assert result.iloc[0]["year"] == "2023"
        assert result.iloc[0]["count"] == 1
        assert result.iloc[1]["year"] == "2024"
        assert result.iloc[1]["count"] == 2

    def test_count_by_quarter(self) -> None:
        """count_by('quarter') groups by quarter."""
        ds = self._make_dataset(
            [
                _utc(2024, 1, 15),
                _utc(2024, 4, 10),
                _utc(2024, 4, 20),
            ]
        )
        result = ds.count_by("quarter")
        assert list(result.columns) == ["quarter", "count"]
        assert result.iloc[0]["quarter"] == "2024-Q1"
        assert result.iloc[1]["quarter"] == "2024-Q2"
        assert result.iloc[1]["count"] == 2

    def test_count_by_excludes_rejected_by_default(self) -> None:
        """count_by() excludes rejected rows by default."""
        ds = self._make_dataset(
            [_utc(2024, 1, 1), _utc(2024, 1, 2)],
            statuses=["analyzed", "rejected"],
        )
        result = ds.count_by("month")
        assert result.iloc[0]["count"] == 1

    def test_count_by_includes_rejected_when_requested(self) -> None:
        """count_by(rejected=True) includes rejected rows."""
        ds = self._make_dataset(
            [_utc(2024, 1, 1), _utc(2024, 1, 2)],
            statuses=["analyzed", "rejected"],
        )
        result = ds.count_by("month", rejected=True)
        assert result.iloc[0]["count"] == 2

    def test_count_by_with_date_range(self) -> None:
        """count_by() respects start/end date filtering."""
        ds = self._make_dataset(
            [
                _utc(2024, 1, 1),
                _utc(2024, 6, 1),
                _utc(2024, 12, 1),
            ]
        )
        result = ds.count_by("month", start="2024-06-01", end="2024-12-01")
        assert len(result) == 1
        assert result.iloc[0]["month"] == "2024-06"

    def test_count_by_empty_result(self) -> None:
        """count_by() with no matching rows returns empty DataFrame with correct columns."""
        ds = self._make_dataset([_utc(2024, 1, 1)])
        result = ds.count_by("month", start="2025-01-01")
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["month", "count"]
        assert len(result) == 0

    def test_count_by_invalid_period_raises(self) -> None:
        """count_by() raises ValueError for invalid period."""
        ds = self._make_dataset([_utc(2024, 1, 1)])
        with pytest.raises(ValueError, match="period"):
            ds.count_by("hourly")

    def test_count_by_sorted_by_period(self) -> None:
        """count_by() returns results sorted chronologically."""
        ds = self._make_dataset(
            [
                _utc(2024, 12, 1),
                _utc(2024, 1, 1),
                _utc(2024, 6, 1),
            ]
        )
        result = ds.count_by("month")
        periods = list(result["month"])
        assert periods == sorted(periods)


class TestProvenanceDateRangeMerge:
    def test_merge_takes_min_max_across_datasets(self) -> None:
        """merge() computes min(date_min) and max(date_max)."""
        ds1 = VulnDataset()
        ds1._provenance["date_min"] = "2024-01-01T00:00:00+00:00"
        ds1._provenance["date_max"] = "2024-06-30T00:00:00+00:00"
        ds1.set_column("cve_id", np.array(["CVE-2024-0001"], dtype=object))

        ds2 = VulnDataset()
        ds2._provenance["date_min"] = "2024-07-01T00:00:00+00:00"
        ds2._provenance["date_max"] = "2024-12-31T00:00:00+00:00"
        ds2.set_column("cve_id", np.array(["CVE-2024-0002"], dtype=object))

        merged = VulnDataset.merge([ds1, ds2])
        assert merged._provenance["date_min"] == "2024-01-01T00:00:00+00:00"
        assert merged._provenance["date_max"] == "2024-12-31T00:00:00+00:00"

    def test_merge_handles_missing_date_range(self) -> None:
        """merge() handles datasets without date_min/date_max (old format)."""
        ds1 = VulnDataset()
        ds1._provenance["date_min"] = "2024-01-01T00:00:00+00:00"
        ds1._provenance["date_max"] = "2024-06-30T00:00:00+00:00"
        ds1.set_column("cve_id", np.array(["CVE-2024-0001"], dtype=object))

        ds2 = VulnDataset()  # No date_min/date_max (old provenance)
        ds2.set_column("cve_id", np.array(["CVE-2024-0002"], dtype=object))

        merged = VulnDataset.merge([ds1, ds2])
        assert merged._provenance["date_min"] == "2024-01-01T00:00:00+00:00"
        assert merged._provenance["date_max"] == "2024-06-30T00:00:00+00:00"


class TestErrorWrapping:
    """Tests for consistent exception handling at API boundaries."""

    def test_from_pandas_wraps_error_as_extractor_error(self) -> None:
        """from_pandas() wraps internal errors as ExtractorError."""
        with pytest.raises(ExtractorError):
            VulnDataset.from_pandas(None)  # type: ignore[arg-type]

    def test_from_pandas_suppresses_exception_chain(self) -> None:
        """from_pandas() uses from None to suppress chained tracebacks."""
        with pytest.raises(ExtractorError) as exc_info:
            VulnDataset.from_pandas(None)  # type: ignore[arg-type]
        assert exc_info.value.__cause__ is None

    def test_from_polars_wraps_error_as_extractor_error(self) -> None:
        """from_polars() wraps internal errors as ExtractorError."""
        with pytest.raises(ExtractorError):
            VulnDataset.from_polars(None)  # type: ignore[arg-type]

    def test_from_polars_suppresses_exception_chain(self) -> None:
        """from_polars() uses from None to suppress chained tracebacks."""
        with pytest.raises(ExtractorError) as exc_info:
            VulnDataset.from_polars(None)  # type: ignore[arg-type]
        assert exc_info.value.__cause__ is None

    def test_to_parquet_wraps_error_as_loader_error(self, tmp_path: Path) -> None:
        """to_parquet() wraps internal errors as LoaderError."""
        ds = VulnDataset()
        ds.set_column("cve_id", np.array(["CVE-1"], dtype=object))
        # Create a directory at the target path so write fails
        target = tmp_path / "blocker.parquet"
        target.mkdir()
        with pytest.raises(LoaderError):
            ds.to_parquet(target)

    def test_to_parquet_suppresses_exception_chain(self, tmp_path: Path) -> None:
        """to_parquet() uses from None to suppress chained tracebacks."""
        ds = VulnDataset()
        ds.set_column("cve_id", np.array(["CVE-1"], dtype=object))
        target = tmp_path / "blocker.parquet"
        target.mkdir()
        with pytest.raises(LoaderError) as exc_info:
            ds.to_parquet(target)
        assert exc_info.value.__cause__ is None

    def test_to_json_wraps_error_as_loader_error(self, tmp_path: Path) -> None:
        """to_json() wraps internal errors as LoaderError."""
        ds = VulnDataset()
        ds.set_column("cve_id", np.array(["CVE-1"], dtype=object))
        target = tmp_path / "blocker.json"
        target.mkdir()
        with pytest.raises(LoaderError):
            ds.to_json(target)

    def test_to_json_suppresses_exception_chain(self, tmp_path: Path) -> None:
        """to_json() uses from None to suppress chained tracebacks."""
        ds = VulnDataset()
        ds.set_column("cve_id", np.array(["CVE-1"], dtype=object))
        target = tmp_path / "blocker.json"
        target.mkdir()
        with pytest.raises(LoaderError) as exc_info:
            ds.to_json(target)
        assert exc_info.value.__cause__ is None
