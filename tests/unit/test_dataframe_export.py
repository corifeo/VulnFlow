"""
DataFrame Export Tests
Tests: to_pandas(), to_polars() conversions

Layer: Load (DataFrame export)
Dependencies: dataset, pandas, polars
"""

from __future__ import annotations

import pandas as pd

from vulnflow.dataset import VulnDataset


def test_to_pandas_returns_dataframe():
    """to_pandas() returns pandas DataFrame."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")
    df = dataset.to_pandas()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3


def test_to_pandas_has_basic_columns():
    """to_pandas() includes basic columns."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")
    df = dataset.to_pandas()

    assert "cve_id" in df.columns
    assert "nvd_published_at" in df.columns
    assert "description" in df.columns
    assert list(df["cve_id"]) == ["CVE-2024-0001", "CVE-2024-0002", "CVE-2024-0003"]


def test_to_polars_returns_dataframe():
    """to_polars() returns polars DataFrame."""
    import polars as pl

    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")
    df = dataset.to_polars()

    assert isinstance(df, pl.DataFrame)
    assert len(df) == 3


def test_to_polars_has_basic_columns():
    """to_polars() includes basic columns."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")
    df = dataset.to_polars()

    assert "cve_id" in df.columns
    assert "nvd_published_at" in df.columns
    assert "description" in df.columns
    assert df["cve_id"].to_list() == ["CVE-2024-0001", "CVE-2024-0002", "CVE-2024-0003"]


def test_to_pandas_flattens_cvss():
    """to_pandas() with CVSSFlattenTransform produces flattened CVSS columns."""
    from vulnflow.transforms.flatten import CVSSFlattenTransform

    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")
    CVSSFlattenTransform().transform(dataset)
    df = dataset.to_pandas()

    # Flattened CVSS columns exist
    assert "cvss_base_score" in df.columns
    assert "cvss_base_severity" in df.columns
    assert "cvss_version" in df.columns
    assert "cvss_vector_string" in df.columns
    assert "cvss_source" in df.columns

    # First record (CVE-2024-0001) has CVSS 9.8 CRITICAL
    assert df.loc[0, "cvss_base_score"] == 9.8
    assert df.loc[0, "cvss_base_severity"] == "CRITICAL"

    # Second record (CVE-2024-0002) has no CVSS - should be None
    assert pd.isna(df.loc[1, "cvss_base_score"])


def test_to_polars_flattens_cvss():
    """to_polars() with CVSSFlattenTransform produces flattened CVSS columns."""
    from vulnflow.transforms.flatten import CVSSFlattenTransform

    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")
    CVSSFlattenTransform().transform(dataset)
    df = dataset.to_polars()

    # Flattened CVSS columns exist
    assert "cvss_base_score" in df.columns
    assert "cvss_base_severity" in df.columns

    # First record has CVSS
    assert df["cvss_base_score"][0] == 9.8
    assert df["cvss_base_severity"][0] == "CRITICAL"
