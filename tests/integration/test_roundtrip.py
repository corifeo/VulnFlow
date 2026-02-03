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


def test_parquet_roundtrip_preserves_dataframe(tmp_path):
    """Data survives NVD → Parquet → DataFrame roundtrip."""
    import math

    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    output_path = tmp_path / "roundtrip.parquet"
    dataset.to_parquet(output_path)

    dataset2 = VulnDataset.from_parquet(output_path)

    # Compare DataFrames - check columns and values match
    df1 = dataset.to_pandas()
    df2 = dataset2.to_pandas()

    # Same columns
    assert set(df1.columns) == set(df2.columns)

    # Same row count
    assert len(df1) == len(df2)

    # CVE IDs match
    assert list(df1["cve_id"]) == list(df2["cve_id"])

    # CVSS scores match (handle None/NaN equivalence)
    scores1 = df1["cvss_base_score"].tolist()
    scores2 = df2["cvss_base_score"].tolist()
    for s1, s2 in zip(scores1, scores2, strict=True):
        if s1 is None or (isinstance(s1, float) and math.isnan(s1)):
            assert s2 is None or (isinstance(s2, float) and math.isnan(s2))
        else:
            assert s1 == s2


def test_pandas_polars_consistency():
    """Pandas and Polars export have same data."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    df_pandas = dataset.to_pandas()
    df_polars = dataset.to_polars()

    # Compare lengths
    assert len(df_pandas) == len(df_polars)

    # Compare CVE IDs
    assert list(df_pandas["cve_id"]) == df_polars["cve_id"].to_list()

    # Compare CVSS scores
    pandas_scores = df_pandas["cvss_base_score"].tolist()
    polars_scores = df_polars["cvss_base_score"].to_list()

    # First record has CVSS 9.8
    assert pandas_scores[0] == polars_scores[0] == 9.8

    # Second record has no CVSS (None)
    assert pandas_scores[1] is None
    assert polars_scores[1] is None


# JSON round-trip tests


def test_json_roundtrip_preserves_count(tmp_path):
    """JSON export/import preserves record count."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    output_path = tmp_path / "out.json"
    dataset.to_json(output_path)

    dataset2 = VulnDataset.from_json(output_path)
    assert len(dataset) == len(dataset2)
    assert len(dataset2) == 3


def test_json_roundtrip_preserves_cve_ids(tmp_path):
    """JSON round-trip preserves CVE IDs."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    output_path = tmp_path / "out.json"
    dataset.to_json(output_path)

    dataset2 = VulnDataset.from_json(output_path)

    # Compare CVE IDs
    cve_ids_1 = list(dataset._data["cve_id"])
    cve_ids_2 = list(dataset2._data["cve_id"])
    assert cve_ids_1 == cve_ids_2


def test_json_roundtrip_preserves_columns(tmp_path):
    """JSON round-trip preserves all columns."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    output_path = tmp_path / "out.json"
    dataset.to_json(output_path)

    dataset2 = VulnDataset.from_json(output_path)

    # Same columns
    assert set(dataset.columns) == set(dataset2.columns)


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


# Pandas round-trip tests


def test_pandas_roundtrip_preserves_count():
    """Pandas export/import preserves record count."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    df = dataset.to_pandas()
    dataset2 = VulnDataset.from_pandas(df)

    assert len(dataset) == len(dataset2)
    assert len(dataset2) == 3


def test_pandas_roundtrip_preserves_columns():
    """Pandas round-trip preserves all columns (with include_nested=True)."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    df = dataset.to_pandas(include_nested=True)
    dataset2 = VulnDataset.from_pandas(df)

    # Same columns
    assert set(dataset.columns) == set(dataset2.columns)


def test_pandas_roundtrip_preserves_cve_ids():
    """Pandas round-trip preserves CVE IDs."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    df = dataset.to_pandas()
    dataset2 = VulnDataset.from_pandas(df)

    # Compare CVE IDs
    cve_ids_1 = list(dataset._data["cve_id"])
    cve_ids_2 = list(dataset2._data["cve_id"])
    assert cve_ids_1 == cve_ids_2


# Polars round-trip tests


def test_polars_roundtrip_preserves_count():
    """Polars export/import preserves record count."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    df = dataset.to_polars()
    dataset2 = VulnDataset.from_polars(df)

    assert len(dataset) == len(dataset2)
    assert len(dataset2) == 3


def test_polars_roundtrip_preserves_columns():
    """Polars round-trip preserves all columns (with include_nested=True)."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    df = dataset.to_polars(include_nested=True)
    dataset2 = VulnDataset.from_polars(df)

    # Same columns
    assert set(dataset.columns) == set(dataset2.columns)


def test_polars_roundtrip_preserves_cve_ids():
    """Polars round-trip preserves CVE IDs."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    df = dataset.to_polars()
    dataset2 = VulnDataset.from_polars(df)

    # Compare CVE IDs
    cve_ids_1 = list(dataset._data["cve_id"])
    cve_ids_2 = list(dataset2._data["cve_id"])
    assert cve_ids_1 == cve_ids_2
