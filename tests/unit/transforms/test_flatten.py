"""
Flatten Transforms - CVSS, CWE, CPE, EPSS flattening and DataFrame integration
Tests: Selection logic, output columns, edge cases, DataFrame export

Layer: Transform/Flatten
Dependencies: transforms.flatten.base, dataset
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from vulnflow import VulnDataset
from vulnflow.transforms.flatten import (
    CPEFlattenTransform,
    CVSSFlattenTransform,
    CWEFlattenTransform,
)
from vulnflow.utils import json as vulnflow_json

# =============================================================================
# BaseFlatten Tests
# =============================================================================


def test_base_flatten_is_abstract():
    """BaseFlatten cannot be instantiated directly."""
    from vulnflow.transforms.flatten.base import BaseFlatten

    with pytest.raises(TypeError):
        BaseFlatten()  # type: ignore[abstract]


def test_base_flatten_requires_input_column():
    """Subclass must define input_column."""
    from vulnflow.transforms.flatten.base import BaseFlatten

    class TestFlatten(BaseFlatten):
        name = "test"
        input_column = "test_scores"
        OUTPUT_SCHEMA: list = []
        depends_on: list[str] = []

        def transform(self, dataset):  # type: ignore[override]
            pass

    t = TestFlatten()
    assert t.input_column == "test_scores"


# =============================================================================
# CVSSFlattenTransform Tests
# =============================================================================


def test_cvss_flatten_selects_authoritative():
    """CVSSFlattenTransform selects is_authoritative=True entry."""
    from vulnflow.transforms.flatten.cvss import CVSSFlattenTransform

    ds = VulnDataset()
    ds._data = {
        "cve_id": np.array(["CVE-2021-44228"]),
        "cvss_scores": np.array(
            [
                vulnflow_json.dumps(
                    [
                        {
                            "version": "3.1",
                            "base_score": 9.0,
                            "source": "vendor",
                            "is_authoritative": False,
                        },
                        {
                            "version": "3.1",
                            "base_score": 10.0,
                            "source": "nvd",
                            "is_authoritative": True,
                        },
                    ]
                )
            ]
        ),
    }

    CVSSFlattenTransform().transform(ds)

    assert ds.get_column("cvss_base_score")[0] == 10.0
    assert ds.get_column("cvss_source")[0] == "nvd"


def test_cvss_flatten_parses_vector_string():
    """CVSSFlattenTransform extracts CVSS vector components."""
    from vulnflow.transforms.flatten.cvss import CVSSFlattenTransform

    ds = VulnDataset()
    ds._data = {
        "cve_id": np.array(["CVE-2021-44228"]),
        "cvss_scores": np.array(
            [
                vulnflow_json.dumps(
                    [
                        {
                            "version": "3.1",
                            "vector_string": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H",
                            "base_score": 10.0,
                            "base_severity": "CRITICAL",
                            "source": "nvd",
                            "is_authoritative": True,
                        }
                    ]
                )
            ]
        ),
    }

    CVSSFlattenTransform().transform(ds)

    assert ds.get_column("cvss_attack_vector")[0] == "N"
    assert ds.get_column("cvss_attack_complexity")[0] == "L"
    assert ds.get_column("cvss_privileges_required")[0] == "N"
    assert ds.get_column("cvss_user_interaction")[0] == "N"
    assert ds.get_column("cvss_scope")[0] == "C"
    assert ds.get_column("cvss_confidentiality")[0] == "H"
    assert ds.get_column("cvss_integrity")[0] == "H"
    assert ds.get_column("cvss_availability")[0] == "H"


def test_cvss_flatten_fallback_to_first():
    """CVSSFlattenTransform falls back to first entry when no authoritative."""
    from vulnflow.transforms.flatten.cvss import CVSSFlattenTransform

    ds = VulnDataset()
    ds._data = {
        "cve_id": np.array(["CVE-2021-44228"]),
        "cvss_scores": np.array(
            [
                vulnflow_json.dumps(
                    [
                        {
                            "version": "3.1",
                            "base_score": 9.0,
                            "source": "first",
                            "is_authoritative": False,
                        },
                        {
                            "version": "3.1",
                            "base_score": 8.0,
                            "source": "second",
                            "is_authoritative": False,
                        },
                    ]
                )
            ]
        ),
    }

    CVSSFlattenTransform().transform(ds)

    assert ds.get_column("cvss_base_score")[0] == 9.0
    assert ds.get_column("cvss_source")[0] == "first"


def test_cvss_flatten_handles_empty_array():
    """CVSSFlattenTransform handles empty cvss_scores gracefully."""
    from vulnflow.transforms.flatten.cvss import CVSSFlattenTransform

    ds = VulnDataset()
    ds._data = {
        "cve_id": np.array(["CVE-2021-44228"]),
        "cvss_scores": np.array(["[]"]),
    }

    CVSSFlattenTransform().transform(ds)

    assert ds.get_column("cvss_base_score")[0] is None
    assert ds.get_column("cvss_source")[0] is None


# =============================================================================
# CWEFlattenTransform Tests
# =============================================================================


def test_cwe_primary_extracts_first():
    """CWEFlattenTransform extracts first CWE ID."""
    from vulnflow.transforms.flatten.cwe import CWEFlattenTransform

    ds = VulnDataset()
    ds._data = {
        "cve_id": np.array(["CVE-2021-44228"]),
        "cwes": np.array(
            [
                vulnflow_json.dumps(
                    [
                        {"id": "CWE-502", "name": "Deserialization of Untrusted Data"},
                        {"id": "CWE-20", "name": "Improper Input Validation"},
                    ]
                )
            ]
        ),
    }

    CWEFlattenTransform().transform(ds)

    assert ds.get_column("cwe_id")[0] == "CWE-502"


def test_cwe_primary_handles_empty_array():
    """CWEFlattenTransform handles empty cwes gracefully."""
    from vulnflow.transforms.flatten.cwe import CWEFlattenTransform

    ds = VulnDataset()
    ds._data = {
        "cve_id": np.array(["CVE-2021-44228"]),
        "cwes": np.array(["[]"]),
    }

    CWEFlattenTransform().transform(ds)

    assert ds.get_column("cwe_id")[0] is None


def test_cwe_flatten_extracts_all_enriched_fields() -> None:
    """Test CWEFlattenTransform extracts all enriched CWE fields."""
    dataset = VulnDataset()
    dataset._data = {
        "cve_id": np.array(["CVE-2024-0001"], dtype=object),
        "cwes": np.array(
            [
                json.dumps(
                    [
                        {
                            "id": "CWE-79",
                            "name": "Cross-site Scripting (XSS)",
                            "category_id": "CWE-74",
                            "category_name": "Improper Neutralization",
                            "abstraction_level": "Base",
                        }
                    ]
                )
            ],
            dtype=object,
        ),
    }

    transform = CWEFlattenTransform()
    transform.transform(dataset)

    assert dataset.get_column("cwe_id")[0] == "CWE-79"
    assert dataset.get_column("cwe_name")[0] == "Cross-site Scripting (XSS)"
    assert dataset.get_column("cwe_category_id")[0] == "CWE-74"
    assert dataset.get_column("cwe_category_name")[0] == "Improper Neutralization"
    assert dataset.get_column("cwe_abstraction_level")[0] == "Base"


# =============================================================================
# CPEFlattenTransform Tests
# =============================================================================


def test_cpe_primary_extracts_first():
    """CPEFlattenTransform extracts first CPE by default."""
    from vulnflow.transforms.flatten.cpe import CPEFlattenTransform

    ds = VulnDataset()
    ds._data = {
        "cve_id": np.array(["CVE-2021-44228"]),
        "cpes": np.array(
            [
                vulnflow_json.dumps(
                    [
                        {"vendor": "apache", "product": "log4j", "version": "2.14.1"},
                        {
                            "vendor": "debian",
                            "product": "debian_linux",
                            "version": "10.0",
                        },
                    ]
                )
            ]
        ),
    }

    CPEFlattenTransform().transform(ds)

    assert ds.get_column("cpe_vendor")[0] == "apache"
    assert ds.get_column("cpe_product")[0] == "log4j"
    assert ds.get_column("cpe_version")[0] == "2.14.1"


def test_cpe_primary_handles_empty_array():
    """CPEFlattenTransform handles empty cpes gracefully."""
    from vulnflow.transforms.flatten.cpe import CPEFlattenTransform

    ds = VulnDataset()
    ds._data = {
        "cve_id": np.array(["CVE-2021-44228"]),
        "cpes": np.array(["[]"]),
    }

    CPEFlattenTransform().transform(ds)

    assert ds.get_column("cpe_vendor")[0] is None
    assert ds.get_column("cpe_product")[0] is None


def test_cpe_primary_extracts_fix_version():
    """CPEFlattenTransform extracts fix_version if present."""
    from vulnflow.transforms.flatten.cpe import CPEFlattenTransform

    ds = VulnDataset()
    ds._data = {
        "cve_id": np.array(["CVE-2021-44228"]),
        "cpes": np.array(
            [
                vulnflow_json.dumps(
                    [
                        {
                            "vendor": "apache",
                            "product": "log4j",
                            "version": "2.14.1",
                            "fix_version": "2.17.0",
                        },
                    ]
                )
            ]
        ),
    }

    CPEFlattenTransform().transform(ds)

    assert ds.get_column("cpe_fix_version")[0] == "2.17.0"


# =============================================================================
# EPSSFlattenTransform Tests
# =============================================================================


def test_epss_flatten_selects_most_recent():
    """EPSSFlattenTransform selects entry with latest date."""
    from vulnflow.transforms.flatten.epss import EPSSFlattenTransform

    ds = VulnDataset()
    ds._data = {
        "cve_id": np.array(["CVE-2021-44228"]),
        "epss_scores": np.array(
            [
                vulnflow_json.dumps(
                    [
                        {
                            "probability": 0.5,
                            "percentile": 0.8,
                            "date": "2024-01-01T00:00:00+00:00",
                            "model": 3,
                        },
                        {
                            "probability": 0.9,
                            "percentile": 0.95,
                            "date": "2024-02-01T00:00:00+00:00",
                            "model": 3,
                        },
                    ]
                )
            ]
        ),
    }

    EPSSFlattenTransform().transform(ds)

    assert ds.get_column("epss_probability")[0] == 0.9
    assert ds.get_column("epss_percentile")[0] == 0.95


def test_epss_flatten_handles_empty_array():
    """EPSSFlattenTransform handles empty epss_scores gracefully."""
    from vulnflow.transforms.flatten.epss import EPSSFlattenTransform

    ds = VulnDataset()
    ds._data = {
        "cve_id": np.array(["CVE-2021-44228"]),
        "epss_scores": np.array(["[]"]),
    }

    EPSSFlattenTransform().transform(ds)

    assert ds.get_column("epss_probability")[0] is None
    assert ds.get_column("epss_percentile")[0] is None


def test_epss_flatten_extracts_all_columns():
    """EPSSFlattenTransform extracts all 4 columns."""
    from vulnflow.transforms.flatten.epss import EPSSFlattenTransform

    ds = VulnDataset()
    ds._data = {
        "cve_id": np.array(["CVE-2021-44228"]),
        "epss_scores": np.array(
            [
                vulnflow_json.dumps(
                    [
                        {
                            "probability": 0.9,
                            "percentile": 0.95,
                            "date": "2024-02-01T00:00:00+00:00",
                            "model": 3,
                        },
                    ]
                )
            ]
        ),
    }

    EPSSFlattenTransform().transform(ds)

    assert "epss_probability" in ds.columns
    assert "epss_percentile" in ds.columns
    assert "epss_date" in ds.columns
    assert "epss_model" in ds.columns


# =============================================================================
# DataFrame Integration Tests
# =============================================================================


def test_dataframe_flattens_primary_cwe():
    """CWEFlattenTransform flattens first CWE to cwe_id column."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    # Apply flatten transform
    CWEFlattenTransform().transform(dataset)

    df = dataset.to_pandas()

    # Check column exists
    assert "cwe_id" in df.columns

    # First record should have CWE-94
    assert df["cwe_id"].iloc[0] == "CWE-94"


def test_dataframe_flattens_primary_cpe():
    """CPEFlattenTransform flattens first CPE to cpe_* columns."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    # Apply flatten transform
    CPEFlattenTransform().transform(dataset)

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
    """DataFrame export with include_nested=True preserves full JSON lists."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")
    df = dataset.to_pandas(include_nested=True)  # Full fidelity mode

    # Full lists should exist
    assert "cwes" in df.columns
    assert "cpes" in df.columns
    assert "cvss_scores" in df.columns

    # They should be JSON strings
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
