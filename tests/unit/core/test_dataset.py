"""
Dataset Core - VulnDataset with columnar storage
Tests: VulnDataset creation, __len__, columns

Layer: Core ETL data structure
Dependencies: schema
"""

from __future__ import annotations

import numpy as np
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
    """dataset.to_pandas(include_nested=True) returns all dataset columns."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    df = dataset.to_pandas(include_nested=True)  # Full fidelity

    # Should have nested JSON columns when include_nested=True
    assert "references" in df.columns
    assert "cvss_scores" in df.columns

    # Flattened columns require applying transforms first
    # (cvss_base_score, cwe_id, cpe_vendor etc not present by default)


def test_to_polars_delegates_to_polars_loader():
    """dataset.to_polars(include_nested=True) returns all dataset columns."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    df = dataset.to_polars(include_nested=True)  # Full fidelity

    # Should have nested JSON columns when include_nested=True
    assert "references" in df.columns
    assert "cvss_scores" in df.columns

    # Flattened columns require applying transforms first
    # (cvss_base_score, cwe_id, cpe_vendor etc not present by default)


def test_merge_empty_list():
    """VulnDataset.merge([]) returns empty dataset."""
    result = VulnDataset.merge([])
    assert len(result) == 0


def test_merge_single_dataset():
    """VulnDataset.merge([ds]) returns dataset unchanged."""
    ds = VulnDataset()
    ds.set_column("cve_id", np.array(["CVE-1", "CVE-2"]))

    result = VulnDataset.merge([ds])

    assert len(result) == 2
    assert result is ds  # Same object


def test_merge_multiple_datasets():
    """VulnDataset.merge() concatenates columns from multiple datasets."""
    ds1 = VulnDataset()
    ds1.set_column("cve_id", np.array(["CVE-1", "CVE-2"]))
    ds1.set_column("score", np.array([7.5, 9.0]))

    ds2 = VulnDataset()
    ds2.set_column("cve_id", np.array(["CVE-3"]))
    ds2.set_column("score", np.array([5.0]))

    result = VulnDataset.merge([ds1, ds2])

    assert len(result) == 3
    assert list(result.get_column("cve_id")) == ["CVE-1", "CVE-2", "CVE-3"]
    assert list(result.get_column("score")) == [7.5, 9.0, 5.0]


def test_merge_handles_missing_columns():
    """VulnDataset.merge() fills None for missing columns."""
    ds1 = VulnDataset()
    ds1.set_column("cve_id", np.array(["CVE-1"]))
    ds1.set_column("score", np.array([7.5]))

    ds2 = VulnDataset()
    ds2.set_column("cve_id", np.array(["CVE-2"]))
    # ds2 missing "score" column

    result = VulnDataset.merge([ds1, ds2])

    assert len(result) == 2
    assert result.get_column("score")[0] == 7.5
    assert result.get_column("score")[1] is None


# =============================================================================
# from_parquet Tests
# =============================================================================


def test_from_parquet_accepts_list(tmp_path):
    """VulnDataset.from_parquet() accepts list[str]."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    # Create two parquet files
    dataset1 = VulnDataset()
    dataset1._data = {"cve_id": np.array(["CVE-2024-0001"])}

    dataset2 = VulnDataset()
    dataset2._data = {"cve_id": np.array(["CVE-2024-0002"])}

    file1 = tmp_path / "file1.parquet"
    file2 = tmp_path / "file2.parquet"

    pq.write_table(pa.table(dataset1._data), file1)
    pq.write_table(pa.table(dataset2._data), file2)

    # Load using from_parquet with list
    loaded = VulnDataset.from_parquet([str(file1), str(file2)])

    assert len(loaded) == 2
    cve_ids = list(loaded.get_column("cve_id"))
    assert "CVE-2024-0001" in cve_ids
    assert "CVE-2024-0002" in cve_ids


def test_from_parquet_single_file_still_works(tmp_path):
    """VulnDataset.from_parquet() with single string still works."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    dataset = VulnDataset()
    dataset._data = {"cve_id": np.array(["CVE-2024-0001"])}

    file_path = tmp_path / "single.parquet"
    pq.write_table(pa.table(dataset._data), file_path)

    loaded = VulnDataset.from_parquet(str(file_path))

    assert len(loaded) == 1
    assert loaded.get_column("cve_id")[0] == "CVE-2024-0001"


# =============================================================================
# Metadata Tests
# =============================================================================


def test_update_sources_adds_new_source():
    """update_sources adds source to list without duplicates."""
    import json

    records = [VulnRecord(cve_id="CVE-2024-0001", sources=["NVD"])]
    dataset = dataset_from_records(records)

    dataset.update_sources([0], "KEV")

    sources_col = dataset.get_column("sources")
    sources = json.loads(sources_col[0])
    assert "KEV" in sources
    assert "NVD" in sources
    assert len(sources) == 2


def test_update_sources_no_duplicates():
    """update_sources doesn't add duplicate sources."""
    import json

    records = [VulnRecord(cve_id="CVE-2024-0001", sources=["NVD", "KEV"])]
    dataset = dataset_from_records(records)

    dataset.update_sources([0], "KEV")

    sources_col = dataset.get_column("sources")
    sources = json.loads(sources_col[0])
    kev_count = sources.count("KEV")
    assert kev_count == 1, "KEV should appear only once"


def test_update_sources_updates_timestamp():
    """update_sources updates last_updated_at timestamp."""
    from datetime import datetime

    records = [VulnRecord(cve_id="CVE-2024-0001", sources=["NVD"])]
    dataset = dataset_from_records(records)

    before = datetime.now()
    dataset.update_sources([0], "KEV")
    after = datetime.now()

    updated_at = dataset.get_column("last_updated_at")
    assert updated_at[0] is not None
    assert before <= updated_at[0] <= after


# =============================================================================
# Progress Bar Tests
# =============================================================================


def test_transform_shows_progress_bar():
    """transform() creates progress bar with correct total."""
    import os
    from unittest.mock import MagicMock, patch

    from vulnflow.transforms.computed.cvd_state import CVDStateTransform

    with patch("vulnflow.utils.progress.progress_bar") as mock_pbar:
        mock_pbar.return_value.__enter__ = MagicMock()
        mock_pbar.return_value.__exit__ = MagicMock()
        mock_pbar.return_value.update = MagicMock()
        mock_pbar.return_value.set_postfix_str = MagicMock()

        dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")
        dataset.transform([CVDStateTransform()])

        # Verify progress bar created with correct params
        mock_pbar.assert_called_once()
        call_kwargs = mock_pbar.call_args.kwargs
        assert call_kwargs["total"] == 1
        assert call_kwargs["desc"] == "Transforming dataset"
        assert call_kwargs["unit"] == "transforms"


def test_transform_updates_progress_per_step():
    """transform() updates progress bar for each sync transform."""
    from unittest.mock import MagicMock, patch

    from vulnflow.transforms.computed.cvd_state import CVDStateTransform

    with patch("vulnflow.utils.progress.progress_bar") as mock_pbar:
        mock_instance = MagicMock()
        mock_pbar.return_value.__enter__.return_value = mock_instance
        mock_pbar.return_value.__exit__.return_value = None

        dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")
        dataset.transform([CVDStateTransform()])

        # Verify update called once (1 sync transform)
        assert mock_instance.update.call_count == 1
        mock_instance.update.assert_called_with(1)


def test_transform_sets_current_transform_name():
    """transform() shows current transform name for sync transforms."""
    from unittest.mock import MagicMock, patch

    from vulnflow.transforms.computed.cvd_state import CVDStateTransform

    with patch("vulnflow.utils.progress.progress_bar") as mock_pbar:
        mock_instance = MagicMock()
        mock_pbar.return_value.__enter__.return_value = mock_instance
        mock_pbar.return_value.__exit__.return_value = None

        dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")
        dataset.transform([CVDStateTransform()])

        # Verify transform name shown (only for sync transforms)
        mock_instance.set_postfix_str.assert_called_with("[CVDStateTransform]")


def test_transform_respects_tqdm_disable_env():
    """transform() respects TQDM_DISABLE environment variable."""
    import os
    from unittest.mock import patch

    from vulnflow.transforms.computed.cvd_state import CVDStateTransform

    with patch.dict(os.environ, {"TQDM_DISABLE": "1"}):
        # Should not raise even if tqdm is disabled
        dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")
        dataset.transform([CVDStateTransform()])

        # Just verify it completes without error
        assert True


def test_async_transforms_batch_progress_update():
    """Async transforms update progress in batch after completion."""
    import asyncio
    from unittest.mock import MagicMock, patch

    class MockAsyncTransform:
        name = "MockAsync"
        depends_on: list[str] = []

        async def transform(self, dataset):
            await asyncio.sleep(0.01)

    with patch("vulnflow.utils.progress.progress_bar") as mock_pbar:
        mock_instance = MagicMock()
        mock_pbar.return_value.__enter__.return_value = mock_instance
        mock_pbar.return_value.__exit__.return_value = None

        dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")
        transforms = [MockAsyncTransform() for _ in range(3)]
        dataset.transform(transforms)

        # Verify update called ONCE with count of all async transforms
        assert mock_instance.update.call_count == 1
        mock_instance.update.assert_called_with(3)
