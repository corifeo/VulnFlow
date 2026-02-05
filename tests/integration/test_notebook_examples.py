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
    # One-liner: NVD → pandas DataFrame
    df = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json").to_pandas()

    assert len(df) == 3
    assert "cve_id" in df.columns
    assert "description" in df.columns


def test_notebook_01_save_load_parquet():
    """Notebook 01: Save to parquet and reload."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "cache.parquet"

        # Save
        VulnDataset.from_nvd("tests/fixtures/nvd_sample.json").to_parquet(cache_path)

        # Load
        df = VulnDataset.from_parquet(cache_path).to_pandas()

        assert len(df) == 3


def test_notebook_02_incremental_upsert():
    """Notebook 02: Cache + delta with upsert semantics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "cache.parquet"
        delta_path = Path(tmpdir) / "delta.parquet"

        # Build cache
        VulnDataset.from_nvd("tests/fixtures/nvd_sample.json").to_parquet(cache_path)

        # Build delta (modified + new)
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
        df = VulnDataset.from_parquet([str(cache_path), str(delta_path)]).to_pandas()

        # Verify upsert: 3 original + 1 new - 1 overwritten = 4
        assert len(df) == 4

        # Verify CVE-2024-0001 was updated
        updated = df[df["cve_id"] == "CVE-2024-0001"].iloc[0]
        assert updated["cvss_base_score"] == 10.0
        assert "UPDATED" in updated["description"]

        # Verify new CVE added
        new_cve = df[df["cve_id"] == "CVE-2024-9999"].iloc[0]
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

        # Build enriched cache
        VulnDataset.from_nvd("tests/fixtures/nvd_sample.json").transform(
            [CVDStateTransform(), EventInference(), EventInference()]
        ).to_parquet(enriched_path)

        # Fast reload
        df = VulnDataset.from_parquet(enriched_path).to_pandas()

        # Verify all transforms preserved
        assert "cvd_state" in df.columns
        assert "v_timestamp" in df.columns
        assert len(df) == 3


def test_notebook_03_incremental_with_transforms():
    """Notebook 03: Combine incremental + transforms."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "cache_transformed.parquet"

        # Build transformed cache
        VulnDataset.from_nvd("tests/fixtures/nvd_sample.json").transform(
            [CVDStateTransform(), EventInference()]
        ).to_parquet(cache_path)

        # Load (in production, would merge cache + delta)
        df = VulnDataset.from_parquet(cache_path).to_pandas()

        assert len(df) == 3
        assert "cvd_state" in df.columns
        assert "v_timestamp" in df.columns


def test_all_notebook_patterns_chain():
    """End-to-end: Verify all notebook patterns work together."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Pattern 1: Quick start
        df1 = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json").to_pandas()
        assert len(df1) == 3

        # Pattern 2: Save cache
        cache_path = Path(tmpdir) / "cache.parquet"
        VulnDataset.from_nvd("tests/fixtures/nvd_sample.json").to_parquet(cache_path)

        # Pattern 3: Apply transforms and save
        enriched_path = Path(tmpdir) / "enriched.parquet"
        VulnDataset.from_nvd("tests/fixtures/nvd_sample.json").transform(
            [CVDStateTransform()]
        ).to_parquet(enriched_path)

        # Pattern 4: Load enriched
        df2 = VulnDataset.from_parquet(enriched_path).to_pandas()
        assert "cvd_state" in df2.columns

        # All patterns successful
        assert len(df1) == len(df2)
