"""
VulnDataset.from_parquet() with multi-file support
Tests: Convenience method passes list to ParquetExtractor
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vulnflow.dataset import VulnDataset


def test_from_parquet_accepts_list(tmp_path: Path) -> None:
    """VulnDataset.from_parquet() accepts list[str]."""
    # Create two parquet files
    dataset1 = VulnDataset()
    dataset1._data = {"cve_id": np.array(["CVE-2024-0001"])}

    dataset2 = VulnDataset()
    dataset2._data = {"cve_id": np.array(["CVE-2024-0002"])}

    import pyarrow as pa
    import pyarrow.parquet as pq

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


def test_from_parquet_single_file_still_works(tmp_path: Path) -> None:
    """VulnDataset.from_parquet() with single string still works."""
    dataset = VulnDataset()
    dataset._data = {"cve_id": np.array(["CVE-2024-0001"])}

    import pyarrow as pa
    import pyarrow.parquet as pq

    file_path = tmp_path / "single.parquet"
    pq.write_table(pa.table(dataset._data), file_path)

    loaded = VulnDataset.from_parquet(str(file_path))

    assert len(loaded) == 1
    assert loaded.get_column("cve_id")[0] == "CVE-2024-0001"
