"""
Test notebook examples - verify key patterns work

Tests the code patterns demonstrated in example notebooks
without requiring Jupyter/nbformat dependencies.

Layer: Integration Test
Dependencies: All layers
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from vulnflow import VulnDataset
from vulnflow.transforms.computed.cvd_state import CVDStateTransform
from vulnflow.transforms.inferences.events import EventInference


def test_notebook_01_quick_start():
    """Notebook 01: Quick start one-liner works."""
    # One-liner: NVD → pandas DataFrame (default filters exclude only rejected)
    df = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json").to_pandas()

    # 5 total: analyzed, awaiting, modified, analyzed(pre-2017), rejected(pre-2017)
    # Default DataFrame filter excludes only rejected → 4
    assert len(df) == 4
    assert "cve_id" in df.columns
    assert "description" in df.columns


def test_notebook_02_incremental_upsert():
    """Notebook 02: Cache + delta with upsert semantics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "cache.parquet"
        delta_path = Path(tmpdir) / "delta.parquet"

        # Build cache (file loaders are lossless)
        VulnDataset.from_nvd("tests/fixtures/nvd_sample.json").to_parquet(cache_path)

        # Build delta (modified + new) — no nvd_status column
        delta_ds = VulnDataset()
        delta_ds._data = {
            "cve_id": np.array(["CVE-2024-0001", "CVE-2024-9999"], dtype=object),
            "description": np.array(
                [
                    "UPDATED: Modified description",
                    "NEW: New vulnerability",
                ],
                dtype=object,
            ),
            "cvss_base_score": np.array([10.0, 7.5], dtype=object),
        }
        delta_ds.to_parquet(delta_path)

        # Load cache + delta (upsert)
        # Use legacy to_pandas() (no filtering) to verify upsert semantics.
        merged = VulnDataset.from_parquet([str(cache_path), str(delta_path)])

        from vulnflow.loaders.dataframe import to_pandas

        df_all = to_pandas(merged)

        # Verify upsert: 5 original + 1 new - 1 overwritten = 6
        assert len(df_all) == 6

        # Verify CVE-2024-0001 was updated
        updated = df_all[df_all["cve_id"] == "CVE-2024-0001"].iloc[0]
        assert updated["cvss_base_score"] == 10.0
        assert "UPDATED" in updated["description"]

        # Verify new CVE added
        new_cve = df_all[df_all["cve_id"] == "CVE-2024-9999"].iloc[0]
        assert "NEW" in new_cve["description"]


def test_notebook_03_transforms():
    """Notebook 03: Apply transforms with progress tracking."""
    # Load and transform
    ds = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json").transform(
        [CVDStateTransform(), EventInference()]
    )

    df = ds.to_pandas()

    # Verify CVD state columns added
    assert "cvd_state" in df.columns
    assert "is_zero_day_exploit" in df.columns

    # Verify event inference columns added (timestamp + confidence)
    assert "v_timestamp" in df.columns
    assert "v_confidence" in df.columns


def test_notebook_03_cache_enriched():
    """Notebook 03: Cache enriched data for fast reloading."""
    with tempfile.TemporaryDirectory() as tmpdir:
        enriched_path = Path(tmpdir) / "enriched.parquet"

        # Build enriched cache (file loaders are lossless)
        VulnDataset.from_nvd("tests/fixtures/nvd_sample.json").transform(
            [CVDStateTransform(), EventInference(), EventInference()]
        ).to_parquet(enriched_path)

        # Fast reload
        df = VulnDataset.from_parquet(enriched_path).to_pandas()

        # Verify all transforms preserved
        assert "cvd_state" in df.columns
        assert "v_timestamp" in df.columns
        # Default filter: 4 of 5 (only rejected excluded)
        assert len(df) == 4
