"""
ParquetExtractor multi-file loading and upsert
Tests: Multi-file loading, upsert semantics, edge cases
"""

from __future__ import annotations

import numpy as np
import pytest

from vulnflow.dataset import VulnDataset
from vulnflow.exceptions import ExtractorError
from vulnflow.extractors.files import ParquetExtractor


def test_parquet_extractor_accepts_list_of_paths(tmp_path):
    """ParquetExtractor.__init__() accepts list[str] paths."""
    paths = ["file1.parquet", "file2.parquet"]
    extractor = ParquetExtractor(paths)

    assert extractor.path == paths


def test_parquet_single_file_still_works(tmp_path):
    """Single file loading still works (backwards compatibility)."""
    # Create single parquet file
    dataset = VulnDataset()
    dataset._data = {
        "cve_id": np.array(["CVE-2024-0001", "CVE-2024-0002"]),
        "title": np.array(["Vuln 1", "Vuln 2"]),
    }

    file_path = tmp_path / "single.parquet"
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table(dataset._data)
    pq.write_table(table, file_path)

    # Load with single path
    extractor = ParquetExtractor(str(file_path))
    loaded = extractor.extract()

    assert len(loaded) == 2
    assert list(loaded.get_column("cve_id")) == ["CVE-2024-0001", "CVE-2024-0002"]


def test_parquet_multi_file_concatenates_records(tmp_path):
    """Multi-file loading concatenates all records."""
    # Create two parquet files with different CVEs
    dataset1 = VulnDataset()
    dataset1._data = {
        "cve_id": np.array(["CVE-2024-0001"]),
        "title": np.array(["Vuln 1"]),
    }

    dataset2 = VulnDataset()
    dataset2._data = {
        "cve_id": np.array(["CVE-2024-0002"]),
        "title": np.array(["Vuln 2"]),
    }

    import pyarrow as pa
    import pyarrow.parquet as pq

    file1 = tmp_path / "file1.parquet"
    file2 = tmp_path / "file2.parquet"

    pq.write_table(pa.table(dataset1._data), file1)
    pq.write_table(pa.table(dataset2._data), file2)

    # Load both files
    extractor = ParquetExtractor([str(file1), str(file2)])
    loaded = extractor.extract()

    assert len(loaded) == 2
    cve_ids = list(loaded.get_column("cve_id"))
    assert "CVE-2024-0001" in cve_ids
    assert "CVE-2024-0002" in cve_ids


def test_parquet_multi_file_upsert_delta_wins(tmp_path):
    """Duplicate CVE IDs: later file wins (upsert semantics)."""
    # Cache: CVE-2024-0001 with title "Old"
    cache = VulnDataset()
    cache._data = {
        "cve_id": np.array(["CVE-2024-0001", "CVE-2024-0002"]),
        "title": np.array(["Old Title", "Cache Only"]),
    }

    # Delta: CVE-2024-0001 with title "New" (updated)
    delta = VulnDataset()
    delta._data = {
        "cve_id": np.array(["CVE-2024-0001", "CVE-2024-0003"]),
        "title": np.array(["New Title", "Delta Only"]),
    }

    import pyarrow as pa
    import pyarrow.parquet as pq

    cache_file = tmp_path / "cache.parquet"
    delta_file = tmp_path / "delta.parquet"

    pq.write_table(pa.table(cache._data), cache_file)
    pq.write_table(pa.table(delta._data), delta_file)

    # Load with upsert (cache then delta)
    extractor = ParquetExtractor([str(cache_file), str(delta_file)])
    loaded = extractor.extract()

    # Should have 3 unique CVEs
    assert len(loaded) == 3

    # CVE-2024-0001 should have "New Title" (delta wins)
    cve_ids = loaded.get_column("cve_id")
    titles = loaded.get_column("title")

    idx_0001 = np.where(cve_ids == "CVE-2024-0001")[0][0]
    assert titles[idx_0001] == "New Title"

    # CVE-2024-0002 should exist (cache only)
    assert "CVE-2024-0002" in cve_ids

    # CVE-2024-0003 should exist (delta only)
    assert "CVE-2024-0003" in cve_ids


def test_parquet_multi_file_missing_file_raises(tmp_path):
    """Missing file in list raises ExtractorError."""
    file1 = tmp_path / "exists.parquet"

    # Create one valid file
    dataset = VulnDataset()
    dataset._data = {"cve_id": np.array(["CVE-2024-0001"])}

    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.table(dataset._data), file1)

    # Try to load with missing file
    extractor = ParquetExtractor([str(file1), "nonexistent.parquet"])

    with pytest.raises(FileNotFoundError, match="nonexistent.parquet"):
        extractor.extract()


def test_parquet_empty_list_returns_empty_dataset():
    """Empty file list returns empty VulnDataset."""
    extractor = ParquetExtractor([])
    loaded = extractor.extract()

    assert len(loaded) == 0
    assert loaded.columns == []


def test_parquet_multi_file_no_cve_id_raises(tmp_path):
    """Multi-file without cve_id column raises clear error."""
    # Create two files without cve_id column
    dataset1 = VulnDataset()
    dataset1._data = {"title": np.array(["Vuln 1"])}

    dataset2 = VulnDataset()
    dataset2._data = {"title": np.array(["Vuln 2"])}

    import pyarrow as pa
    import pyarrow.parquet as pq

    file1 = tmp_path / "file1.parquet"
    file2 = tmp_path / "file2.parquet"

    pq.write_table(pa.table(dataset1._data), file1)
    pq.write_table(pa.table(dataset2._data), file2)

    extractor = ParquetExtractor([str(file1), str(file2)])

    with pytest.raises(ExtractorError, match="cve_id.*required.*merge"):
        extractor.extract()


def test_parquet_multi_file_schema_mismatch_fills_none(tmp_path):
    """Schema mismatch: missing columns filled with None."""
    # File 1: has cve_id, title
    dataset1 = VulnDataset()
    dataset1._data = {
        "cve_id": np.array(["CVE-2024-0001"]),
        "title": np.array(["Vuln 1"]),
    }

    # File 2: has cve_id, description (missing title)
    dataset2 = VulnDataset()
    dataset2._data = {
        "cve_id": np.array(["CVE-2024-0002"]),
        "description": np.array(["Desc 2"]),
    }

    import pyarrow as pa
    import pyarrow.parquet as pq

    file1 = tmp_path / "file1.parquet"
    file2 = tmp_path / "file2.parquet"

    pq.write_table(pa.table(dataset1._data), file1)
    pq.write_table(pa.table(dataset2._data), file2)

    extractor = ParquetExtractor([str(file1), str(file2)])
    loaded = extractor.extract()

    # Should have both columns (union of schemas)
    assert "cve_id" in loaded.columns
    assert "title" in loaded.columns
    assert "description" in loaded.columns

    # First record has title, no description
    titles = loaded.get_column("title")
    descriptions = loaded.get_column("description")

    assert titles[0] == "Vuln 1"
    assert descriptions[0] is None

    # Second record has description, no title
    assert titles[1] is None
    assert descriptions[1] == "Desc 2"


def test_parquet_multi_file_preserves_column_order(tmp_path):
    """Multi-file loading preserves first file's column order."""
    # File 1: columns in order: cve_id, title, score
    dataset1 = VulnDataset()
    dataset1._data = {
        "cve_id": np.array(["CVE-2024-0001"]),
        "title": np.array(["Vuln 1"]),
        "score": np.array([7.5]),
    }

    # File 2: different column order: severity, cve_id, tags
    # (note: also has different columns to test appending)
    dataset2 = VulnDataset()
    dataset2._data = {
        "severity": np.array(["HIGH"]),
        "cve_id": np.array(["CVE-2024-0002"]),
        "tags": np.array(["exploit"]),
    }

    import pyarrow as pa
    import pyarrow.parquet as pq

    file1 = tmp_path / "file1.parquet"
    file2 = tmp_path / "file2.parquet"

    pq.write_table(pa.table(dataset1._data), file1)
    pq.write_table(pa.table(dataset2._data), file2)

    extractor = ParquetExtractor([str(file1), str(file2)])
    loaded = extractor.extract()

    # Should preserve file1's order (cve_id, title, score) and append new columns
    expected_order = ["cve_id", "title", "score", "severity", "tags"]
    assert loaded.columns == expected_order


def test_parquet_multi_file_repeated_loads_stable(tmp_path):
    """Repeated multi-file loads produce identical column order."""
    # Create two files
    dataset1 = VulnDataset()
    dataset1._data = {
        "cve_id": np.array(["CVE-2024-0001"]),
        "a": np.array([1]),
        "b": np.array([2]),
    }

    dataset2 = VulnDataset()
    dataset2._data = {
        "cve_id": np.array(["CVE-2024-0002"]),
        "c": np.array([3]),
        "d": np.array([4]),
    }

    import pyarrow as pa
    import pyarrow.parquet as pq

    file1 = tmp_path / "file1.parquet"
    file2 = tmp_path / "file2.parquet"

    pq.write_table(pa.table(dataset1._data), file1)
    pq.write_table(pa.table(dataset2._data), file2)

    # Load multiple times
    extractor = ParquetExtractor([str(file1), str(file2)])
    loaded1 = extractor.extract()
    loaded2 = extractor.extract()
    loaded3 = extractor.extract()

    # All loads should have identical column order
    assert loaded1.columns == loaded2.columns
    assert loaded2.columns == loaded3.columns
    assert loaded1.columns == ["cve_id", "a", "b", "c", "d"]
