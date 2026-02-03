"""
File Loaders - Tests for Parquet loader

Tests: ParquetLoader class
Layer: Load
Dependencies: dataset
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from vulnflow.dataset import VulnDataset
from vulnflow.loaders.files import ParquetLoader


def test_parquet_loader_writes_file():
    """ParquetLoader.load() writes Parquet file."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "output.parquet"
        loader = ParquetLoader(output_path)

        result = loader.load(dataset)

        assert output_path.exists()
        assert result["record_count"] == len(dataset)


def test_parquet_loader_roundtrip_preserves_data():
    """ParquetLoader roundtrip preserves all data."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "output.parquet"
        loader = ParquetLoader(output_path)

        # Write
        loader.load(dataset)

        # Read back
        restored = VulnDataset.from_parquet(output_path)

        # Verify
        assert len(restored) == len(dataset)
        assert set(restored.columns) == set(dataset.columns)

        # Check CVE IDs preserved
        original_cves = dataset.get_column("cve_id")
        restored_cves = restored.get_column("cve_id")
        assert list(original_cves) == list(restored_cves)


def test_parquet_loader_creates_parent_directories():
    """ParquetLoader creates parent directories if needed."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "nested" / "dir" / "output.parquet"
        loader = ParquetLoader(output_path)

        loader.load(dataset)

        assert output_path.exists()
