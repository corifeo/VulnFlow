"""
DataFrame Flattening Tests - Verify CWE/CPE/CVSS flattening

Layer: Unit (Dataset)
Dependencies: schema, dataset
"""

from __future__ import annotations

from vulnflow import VulnDataset


def test_dataframe_flattens_primary_cwe():
    """DataFrame export flattens first CWE to primary_cwe_* columns."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")
    df = dataset.to_pandas()

    # Check columns exist
    assert "primary_cwe_id" in df.columns
    assert "primary_cwe_name" in df.columns

    # First record should have CWE-94
    assert df["primary_cwe_id"].iloc[0] == "CWE-94"


def test_dataframe_flattens_primary_cpe():
    """DataFrame export flattens first CPE to primary_* columns."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")
    df = dataset.to_pandas()

    # Check columns exist
    assert "primary_vendor" in df.columns
    assert "primary_product" in df.columns
    assert "primary_version" in df.columns

    # First record should have example/product
    assert df["primary_vendor"].iloc[0] == "example"
    assert df["primary_product"].iloc[0] == "product"


def test_dataframe_flattens_cvss():
    """DataFrame export flattens CVSS to scalar columns."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")
    df = dataset.to_pandas()

    # Check columns exist
    assert "cvss_base_score" in df.columns
    assert "cvss_base_severity" in df.columns
    assert "cvss_version" in df.columns
    assert "cvss_source" in df.columns

    # First record should have CVSS 9.8 from nvd
    assert df["cvss_base_score"].iloc[0] == 9.8
    assert df["cvss_source"].iloc[0] == "nvd"


def test_dataframe_preserves_full_lists():
    """DataFrame export preserves full JSON lists alongside flattened columns when include_nested=True."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")
    df = dataset.to_pandas(include_nested=True)

    # Full lists should still exist
    assert "cwes" in df.columns
    assert "cpes" in df.columns
    assert "cvss_scores" in df.columns

    # They should be JSON strings
    import json

    cwes = json.loads(df["cwes"].iloc[0])
    assert isinstance(cwes, list)
    assert len(cwes) > 0
