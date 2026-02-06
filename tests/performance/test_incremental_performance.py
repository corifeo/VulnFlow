"""
Performance test for incremental workflow
Tests: Multi-file merge performance at scale
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from vulnflow.dataset import VulnDataset


@pytest.mark.slow
def test_large_dataset_merge_performance(tmp_path: Path) -> None:
    """Merge 300K cache + 2K delta completes in <2s."""
    # Create large cache (300K records)
    cache = VulnDataset()
    cache._data = {
        "cve_id": np.array([f"CVE-2024-{i:06d}" for i in range(300_000)]),
        "title": np.array([f"Vuln {i}" for i in range(300_000)]),
    }

    cache_path = tmp_path / "cache.parquet"

    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.table(cache._data), cache_path)

    # Create delta (2K records, 1K updates + 1K new)
    delta_cve_ids = [
        f"CVE-2024-{i:06d}" for i in range(150_000, 151_000)
    ] + [  # 1K updates
        f"CVE-2024-{i:06d}" for i in range(300_000, 301_000)
    ]  # 1K new

    delta = VulnDataset()
    delta._data = {
        "cve_id": np.array(delta_cve_ids),
        "title": np.array([f"Updated {i}" for i in range(2000)]),
    }

    delta_path = tmp_path / "delta.parquet"
    pq.write_table(pa.table(delta._data), delta_path)

    # Time the merge
    start = time.time()
    merged = VulnDataset.from_parquet([str(cache_path), str(delta_path)])
    duration = time.time() - start

    # Verify correctness
    assert len(merged) == 301_000  # 300K + 1K new

    # Performance target: <2s
    assert duration < 2.0, f"Merge took {duration:.2f}s, expected <2s"

    print(f"\nMerged 300K + 2K in {duration:.2f}s")
