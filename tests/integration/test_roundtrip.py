"""
Round-trip Integration Tests
Tests: NVD → Parquet → Load preserves data

Layer: Integration testing
Dependencies: dataset, extractors, loaders
"""

from __future__ import annotations

from vulnflow.dataset import VulnDataset


def test_parquet_roundtrip_preserves_count(tmp_path):
    """Parquet export/import preserves record count."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    output_path = tmp_path / "out.parquet"
    dataset.to_parquet(output_path)

    dataset2 = VulnDataset.from_parquet(output_path)
    assert len(dataset) == len(dataset2)
    assert len(dataset2) == 3


def test_parquet_roundtrip_preserves_cve_ids(tmp_path):
    """Parquet round-trip preserves CVE IDs."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    output_path = tmp_path / "out.parquet"
    dataset.to_parquet(output_path)

    dataset2 = VulnDataset.from_parquet(output_path)

    # Compare CVE IDs
    cve_ids_1 = list(dataset._data["cve_id"])
    cve_ids_2 = list(dataset2._data["cve_id"])
    assert cve_ids_1 == cve_ids_2
