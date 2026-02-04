"""
Round-trip Integration Tests
Tests: NVD → Format → Load preserves data

Layer: Integration testing
Dependencies: dataset, extractors, loaders
"""

from __future__ import annotations

import pytest

from vulnflow.dataset import VulnDataset


@pytest.mark.parametrize(
    "format_name,to_method,from_method,extension",
    [
        ("parquet", "to_parquet", "from_parquet", ".parquet"),
        ("json", "to_json", "from_json", ".json"),
    ],
)
def test_file_roundtrip_preserves_data(tmp_path, format_name, to_method, from_method, extension):
    """File export/import roundtrip preserves count, CVE IDs, and columns."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    # Export
    output_path = tmp_path / f"out{extension}"
    getattr(dataset, to_method)(output_path)

    # Import
    dataset2 = getattr(VulnDataset, from_method)(output_path)

    # Check count
    assert len(dataset) == len(dataset2) == 3

    # Check CVE IDs preserved
    cve_ids_1 = list(dataset._data["cve_id"])
    cve_ids_2 = list(dataset2._data["cve_id"])
    assert cve_ids_1 == cve_ids_2

    # Check columns preserved
    assert set(dataset.columns) == set(dataset2.columns)


@pytest.mark.parametrize(
    "format_name,to_method,from_method",
    [
        ("pandas", "to_pandas", "from_pandas"),
        ("polars", "to_polars", "from_polars"),
    ],
)
def test_dataframe_roundtrip_preserves_data(format_name, to_method, from_method):
    """DataFrame export/import roundtrip preserves count, CVE IDs, and columns."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    # Export (with full fidelity)
    df = getattr(dataset, to_method)(include_nested=True)

    # Import
    dataset2 = getattr(VulnDataset, from_method)(df)

    # Check count
    assert len(dataset) == len(dataset2) == 3

    # Check CVE IDs preserved
    cve_ids_1 = list(dataset._data["cve_id"])
    cve_ids_2 = list(dataset2._data["cve_id"])
    assert cve_ids_1 == cve_ids_2

    # Check columns preserved
    assert set(dataset.columns) == set(dataset2.columns)


def test_parquet_roundtrip_preserves_dataframe(tmp_path):
    """Data survives NVD -> Parquet -> DataFrame roundtrip with nested columns."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    output_path = tmp_path / "roundtrip.parquet"
    dataset.to_parquet(output_path)

    dataset2 = VulnDataset.from_parquet(output_path)

    # Compare DataFrames with include_nested=True for full fidelity
    df1 = dataset.to_pandas(include_nested=True)
    df2 = dataset2.to_pandas(include_nested=True)

    # Same columns
    assert set(df1.columns) == set(df2.columns)

    # Same row count
    assert len(df1) == len(df2)

    # CVE IDs match
    assert list(df1["cve_id"]) == list(df2["cve_id"])

    # Nested JSON columns preserved (cvss_scores array)
    assert "cvss_scores" in df1.columns
    assert "cvss_scores" in df2.columns


def test_pandas_polars_consistency():
    """Pandas and Polars export have same data (both with same include_nested setting)."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    # Compare with default settings (exclude nested)
    df_pandas = dataset.to_pandas()
    df_polars = dataset.to_polars()

    # Compare lengths
    assert len(df_pandas) == len(df_polars)

    # Compare CVE IDs
    assert list(df_pandas["cve_id"]) == df_polars["cve_id"].to_list()

    # Both should have same columns
    assert set(df_pandas.columns) == set(df_polars.columns)


def test_json_vs_parquet_fidelity(tmp_path):
    """Compare JSON vs Parquet round-trip fidelity."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    # JSON round-trip
    json_path = tmp_path / "out.json"
    dataset.to_json(json_path)
    dataset_json = VulnDataset.from_json(json_path)

    # Parquet round-trip
    parquet_path = tmp_path / "out.parquet"
    dataset.to_parquet(parquet_path)
    dataset_parquet = VulnDataset.from_parquet(parquet_path)

    # Both preserve structure
    assert len(dataset_json) == len(dataset_parquet) == len(dataset)
    assert (
        set(dataset_json.columns)
        == set(dataset_parquet.columns)
        == set(dataset.columns)
    )

    # CVE IDs match
    assert list(dataset_json._data["cve_id"]) == list(dataset_parquet._data["cve_id"])
