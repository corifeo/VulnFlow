"""
DataFrame Flattening Tests - Verify CWE/CPE/CVSS flattening via transforms

Layer: Unit (Dataset)
Dependencies: schema, dataset, transforms.flatten

NOTE: Flattening is now done via FlattenTransforms, not automatically during
DataFrame export. Apply transforms before export to get scalar columns.
"""

from __future__ import annotations

from vulnflow import VulnDataset
from vulnflow.transforms.flatten import (
    CPEPrimaryTransform,
    CVSSFlattenTransform,
    CWEPrimaryTransform,
)


def test_dataframe_flattens_primary_cwe():
    """CWEPrimaryTransform flattens first CWE to cwe_* columns."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    # Apply flatten transform
    CWEPrimaryTransform().transform(dataset)

    df = dataset.to_pandas()

    # Check columns exist
    assert "cwe_id" in df.columns
    assert "cwe_name" in df.columns

    # First record should have CWE-94
    assert df["cwe_id"].iloc[0] == "CWE-94"


def test_dataframe_flattens_primary_cpe():
    """CPEPrimaryTransform flattens first CPE to cpe_* columns."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    # Apply flatten transform
    CPEPrimaryTransform().transform(dataset)

    df = dataset.to_pandas()

    # Check columns exist
    assert "cpe_vendor" in df.columns
    assert "cpe_product" in df.columns
    assert "cpe_version" in df.columns

    # First record should have example/product
    assert df["cpe_vendor"].iloc[0] == "example"
    assert df["cpe_product"].iloc[0] == "product"


def test_dataframe_flattens_cvss():
    """CVSSFlattenTransform flattens CVSS to scalar columns."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    # Apply flatten transform
    CVSSFlattenTransform().transform(dataset)

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
    """DataFrame export with include_nested=True preserves full JSON lists (nested arrays)."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")
    df = dataset.to_pandas(include_nested=True)  # Full fidelity mode

    # Full lists should exist
    assert "cwes" in df.columns
    assert "cpes" in df.columns
    assert "cvss_scores" in df.columns

    # They should be JSON strings
    import json

    cwes = json.loads(df["cwes"].iloc[0])
    assert isinstance(cwes, list)
    assert len(cwes) > 0


def test_dataframe_default_excludes_nested():
    """DataFrame export by default excludes nested columns (lossy by default)."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")
    df = dataset.to_pandas()  # Default: include_nested=False

    # Nested JSON columns should be excluded
    assert "cwes" not in df.columns
    assert "cpes" not in df.columns
    assert "cvss_scores" not in df.columns

    # Scalar columns should be present
    assert "cve_id" in df.columns
    assert "description" in df.columns
