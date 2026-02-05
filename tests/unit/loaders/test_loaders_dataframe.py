"""
DataFrame Loaders - Tests for pandas and polars export

Tests: to_pandas() and to_polars() functions
Layer: Load
Dependencies: dataset
"""

from __future__ import annotations

import pandas as pd
import polars as pl

from vulnflow.dataset import VulnDataset
from vulnflow.loaders.dataframe import to_pandas, to_polars


def test_to_pandas_returns_dataframe():
    """to_pandas() returns pandas DataFrame."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    df = to_pandas(dataset)

    assert isinstance(df, pd.DataFrame)


def test_to_pandas_has_all_columns():
    """to_pandas() includes all dataset columns."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    df = to_pandas(dataset)

    # Should have all columns from dataset
    assert set(df.columns) == set(dataset.columns)


def test_to_pandas_correct_row_count():
    """to_pandas() preserves record count."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    df = to_pandas(dataset)

    assert len(df) == len(dataset)


def test_to_polars_returns_dataframe():
    """to_polars() returns polars DataFrame."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    df = to_polars(dataset)

    assert isinstance(df, pl.DataFrame)


def test_to_polars_has_all_columns():
    """to_polars() includes all dataset columns."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    df = to_polars(dataset)

    # Should have all columns from dataset
    assert set(df.columns) == set(dataset.columns)


def test_to_polars_correct_row_count():
    """to_polars() preserves record count."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    df = to_polars(dataset)

    assert len(df) == len(dataset)


def test_pandas_polars_consistency():
    """to_pandas() and to_polars() produce equivalent data."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    pdf = to_pandas(dataset)
    plf = to_polars(dataset)

    # Same columns
    assert set(pdf.columns) == set(plf.columns)

    # Same row count
    assert len(pdf) == len(plf)

    # Same CVE IDs (as a spot check)
    pandas_cves = set(pdf["cve_id"].tolist())
    polars_cves = set(plf["cve_id"].to_list())
    assert pandas_cves == polars_cves


# ============================================================================
# PandasLoader Class Tests
# ============================================================================


def test_pandas_loader_initialization():
    """PandasLoader can be instantiated."""
    from vulnflow.loaders.dataframe import PandasLoader

    loader = PandasLoader()

    assert loader is not None


def test_pandas_loader_load_returns_dataframe():
    """PandasLoader.load() returns pandas DataFrame."""
    from vulnflow.loaders.dataframe import PandasLoader

    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")
    loader = PandasLoader()

    df = loader.load(dataset)

    assert isinstance(df, pd.DataFrame)


def test_pandas_loader_with_flatten_transform():
    """PandasLoader with CVSSFlattenTransform produces flattened columns."""
    from vulnflow.loaders.dataframe import PandasLoader
    from vulnflow.transforms.flatten import CVSSFlattenTransform

    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")
    CVSSFlattenTransform().transform(dataset)
    loader = PandasLoader()  # Default: excludes nested

    df = loader.load(dataset)

    # Should have flattened CVSS columns after transform
    assert "cvss_base_score" in df.columns
    assert "cvss_base_severity" in df.columns
    assert "cvss_version" in df.columns
    assert "cvss_vector_string" in df.columns
    assert "cvss_source" in df.columns

    # Default is lossy - nested columns excluded
    assert "cvss_scores" not in df.columns


def test_pandas_loader_with_flatten_transform_full_fidelity():
    """PandasLoader with include_nested=True includes raw JSON columns."""
    from vulnflow.loaders.dataframe import PandasLoader
    from vulnflow.transforms.flatten import CVSSFlattenTransform

    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")
    CVSSFlattenTransform().transform(dataset)
    loader = PandasLoader(include_nested=True)  # Full fidelity mode

    df = loader.load(dataset)

    # Should have flattened columns
    assert "cvss_base_score" in df.columns
    # Should ALSO have raw JSON column (full fidelity)
    assert "cvss_scores" in df.columns


def test_pandas_loader_with_cwe_transform():
    """PandasLoader with CWEFlattenTransform produces cwe_id column."""
    from vulnflow.loaders.dataframe import PandasLoader
    from vulnflow.transforms.flatten import CWEFlattenTransform

    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")
    CWEFlattenTransform().transform(dataset)
    loader = PandasLoader()

    df = loader.load(dataset)

    assert "cwe_id" in df.columns


def test_pandas_loader_with_cpe_transform():
    """PandasLoader with CPEFlattenTransform produces cpe_* columns."""
    from vulnflow.loaders.dataframe import PandasLoader
    from vulnflow.transforms.flatten import CPEFlattenTransform

    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")
    CPEFlattenTransform().transform(dataset)
    loader = PandasLoader()

    df = loader.load(dataset)

    assert "cpe_vendor" in df.columns
    assert "cpe_product" in df.columns
    assert "cpe_version" in df.columns


def test_pandas_loader_preserves_scalar_columns():
    """PandasLoader keeps scalar columns unchanged."""
    from vulnflow.loaders.dataframe import PandasLoader

    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")
    loader = PandasLoader(include_nested=True)  # Use full fidelity mode

    df = loader.load(dataset)

    # Scalar columns should be present
    assert "cve_id" in df.columns
    assert "description" in df.columns
    assert "nvd_published_at" in df.columns


# ============================================================================
# Task 13: New default API (include_nested=False by default)
# ============================================================================


def test_pandas_loader_excludes_nested_by_default():
    """PandasLoader excludes nested columns by default (lossy by default)."""
    from vulnflow.loaders.dataframe import PandasLoader

    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    loader = PandasLoader()  # Default: include_nested=False
    df = loader.load(dataset)

    # Scalar columns should be present
    assert "cve_id" in df.columns
    assert "description" in df.columns

    # Nested JSON columns should be EXCLUDED by default
    assert "cvss_scores" not in df.columns
    assert "epss_scores" not in df.columns
    assert "cwes" not in df.columns
    assert "cpes" not in df.columns
    assert "references" not in df.columns


def test_pandas_loader_includes_nested_when_requested():
    """PandasLoader includes nested columns when include_nested=True."""
    from vulnflow.loaders.dataframe import PandasLoader

    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    loader = PandasLoader(include_nested=True)
    df = loader.load(dataset)

    # Should include nested JSON columns
    assert "cvss_scores" in df.columns
    assert "cwes" in df.columns
    assert "cpes" in df.columns
    assert "references" in df.columns


def test_polars_loader_excludes_nested_by_default():
    """PolarsLoader excludes nested columns by default (lossy by default)."""
    from vulnflow.loaders.dataframe import PolarsLoader

    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    loader = PolarsLoader()  # Default: include_nested=False
    df = loader.load(dataset)

    # Scalar columns should be present
    assert "cve_id" in df.columns
    assert "description" in df.columns

    # Nested JSON columns should be EXCLUDED by default
    assert "cvss_scores" not in df.columns
    assert "epss_scores" not in df.columns
    assert "cwes" not in df.columns
    assert "cpes" not in df.columns


def test_polars_loader_includes_nested_when_requested():
    """PolarsLoader includes nested columns when include_nested=True."""
    from vulnflow.loaders.dataframe import PolarsLoader

    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    loader = PolarsLoader(include_nested=True)
    df = loader.load(dataset)

    # Should include nested JSON columns
    assert "cvss_scores" in df.columns
    assert "cwes" in df.columns
