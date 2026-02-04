"""
Incremental workflow integration test
Tests: End-to-end notebook workflow with cache + delta
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vulnflow.dataset import VulnDataset


def test_incremental_workflow_cache_plus_delta(tmp_path: Path) -> None:
    """Full incremental workflow: cache + delta with upsert."""
    # Step 1: Initial full load (simulated)
    initial = VulnDataset()
    initial._data = {
        "cve_id": np.array(["CVE-2024-0001", "CVE-2024-0002", "CVE-2024-0003"]),
        "title": np.array(["Vuln 1", "Vuln 2", "Vuln 3"]),
        "cvss_base_score": np.array([7.5, 9.8, 5.0], dtype=float),
    }

    cache_path = tmp_path / "cache.parquet"
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.table(initial._data), cache_path)

    # Step 2: Delta update (modified CVE + new CVE)
    delta = VulnDataset()
    delta._data = {
        "cve_id": np.array(["CVE-2024-0002", "CVE-2024-0004"]),
        "title": np.array(["Vuln 2 UPDATED", "Vuln 4 NEW"]),
        "cvss_base_score": np.array([9.9, 8.5], dtype=float),  # CVE-0002 score updated
    }

    delta_path = tmp_path / "delta.parquet"
    pq.write_table(pa.table(delta._data), delta_path)

    # Step 3: Load merged dataset
    merged = VulnDataset.from_parquet([str(cache_path), str(delta_path)])

    # Verify results
    assert len(merged) == 4  # 3 from cache, 1 new from delta (1 updated)

    cve_ids = merged.get_column("cve_id")
    titles = merged.get_column("title")
    scores = merged.get_column("cvss_base_score")

    # CVE-2024-0001: unchanged from cache
    idx_0001 = np.where(cve_ids == "CVE-2024-0001")[0][0]
    assert titles[idx_0001] == "Vuln 1"
    assert scores[idx_0001] == 7.5

    # CVE-2024-0002: updated from delta
    idx_0002 = np.where(cve_ids == "CVE-2024-0002")[0][0]
    assert titles[idx_0002] == "Vuln 2 UPDATED"
    assert scores[idx_0002] == 9.9

    # CVE-2024-0003: unchanged from cache
    idx_0003 = np.where(cve_ids == "CVE-2024-0003")[0][0]
    assert titles[idx_0003] == "Vuln 3"

    # CVE-2024-0004: new from delta
    idx_0004 = np.where(cve_ids == "CVE-2024-0004")[0][0]
    assert titles[idx_0004] == "Vuln 4 NEW"
    assert scores[idx_0004] == 8.5


def test_incremental_workflow_with_transforms(tmp_path: Path) -> None:
    """Incremental workflow: load merged dataset, apply transforms."""
    # Create cache and delta
    cache = VulnDataset()
    cache._data = {
        "cve_id": np.array(["CVE-2024-0001"]),
        "title": np.array(["Vuln 1"]),
    }

    delta = VulnDataset()
    delta._data = {
        "cve_id": np.array(["CVE-2024-0002"]),
        "title": np.array(["Vuln 2"]),
    }

    import pyarrow as pa
    import pyarrow.parquet as pq

    cache_path = tmp_path / "cache.parquet"
    delta_path = tmp_path / "delta.parquet"

    pq.write_table(pa.table(cache._data), cache_path)
    pq.write_table(pa.table(delta._data), delta_path)

    # Load and transform
    dataset = VulnDataset.from_parquet([str(cache_path), str(delta_path)])

    # Apply transforms (just verify it works, transforms tested elsewhere)
    # Use empty transform list to avoid external API calls
    result = dataset.transform([])

    assert len(result) == 2
    assert result is dataset  # Returns self for chaining


def test_quick_start_one_liner(tmp_path: Path) -> None:
    """Quick start: one line to DataFrame."""
    # Create sample parquet
    dataset = VulnDataset()
    dataset._data = {
        "cve_id": np.array(["CVE-2024-0001", "CVE-2024-0002"]),
        "title": np.array(["Vuln 1", "Vuln 2"]),
    }

    import pyarrow as pa
    import pyarrow.parquet as pq

    file_path = tmp_path / "data.parquet"
    pq.write_table(pa.table(dataset._data), file_path)

    # One-liner: load and convert to pandas
    df = VulnDataset.from_parquet(str(file_path)).to_pandas()

    assert len(df) == 2
    assert list(df["cve_id"]) == ["CVE-2024-0001", "CVE-2024-0002"]
