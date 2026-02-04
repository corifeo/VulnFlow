"""
CVSSFlattenTransform - Select authoritative CVSS and extract columns
Tests: Selection logic, vector parsing, output columns

Layer: Transform/Flatten
Dependencies: transforms.flatten.base
"""

from __future__ import annotations

import numpy as np

from vulnflow.utils import json


def test_cvss_flatten_selects_authoritative():
    """CVSSFlattenTransform selects is_authoritative=True entry."""
    from vulnflow.dataset import VulnDataset
    from vulnflow.transforms.flatten.cvss import CVSSFlattenTransform

    ds = VulnDataset()
    ds._data = {
        "cve_id": np.array(["CVE-2021-44228"]),
        "cvss_scores": np.array([json.dumps([
            {"version": "3.1", "base_score": 9.0, "source": "vendor", "is_authoritative": False},
            {"version": "3.1", "base_score": 10.0, "source": "nvd", "is_authoritative": True},
        ])]),
    }

    CVSSFlattenTransform().transform(ds)

    assert ds.get_column("cvss_base_score")[0] == 10.0
    assert ds.get_column("cvss_source")[0] == "nvd"


def test_cvss_flatten_parses_vector_string():
    """CVSSFlattenTransform extracts CVSS vector components."""
    from vulnflow.dataset import VulnDataset
    from vulnflow.transforms.flatten.cvss import CVSSFlattenTransform

    ds = VulnDataset()
    ds._data = {
        "cve_id": np.array(["CVE-2021-44228"]),
        "cvss_scores": np.array([json.dumps([
            {
                "version": "3.1",
                "vector_string": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H",
                "base_score": 10.0,
                "base_severity": "CRITICAL",
                "source": "nvd",
                "is_authoritative": True,
            }
        ])]),
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
    from vulnflow.dataset import VulnDataset
    from vulnflow.transforms.flatten.cvss import CVSSFlattenTransform

    ds = VulnDataset()
    ds._data = {
        "cve_id": np.array(["CVE-2021-44228"]),
        "cvss_scores": np.array([json.dumps([
            {"version": "3.1", "base_score": 9.0, "source": "first", "is_authoritative": False},
            {"version": "3.1", "base_score": 8.0, "source": "second", "is_authoritative": False},
        ])]),
    }

    CVSSFlattenTransform().transform(ds)

    assert ds.get_column("cvss_base_score")[0] == 9.0
    assert ds.get_column("cvss_source")[0] == "first"


def test_cvss_flatten_handles_empty_array():
    """CVSSFlattenTransform handles empty cvss_scores gracefully."""
    from vulnflow.dataset import VulnDataset
    from vulnflow.transforms.flatten.cvss import CVSSFlattenTransform

    ds = VulnDataset()
    ds._data = {
        "cve_id": np.array(["CVE-2021-44228"]),
        "cvss_scores": np.array(["[]"]),
    }

    CVSSFlattenTransform().transform(ds)

    assert ds.get_column("cvss_base_score")[0] is None
    assert ds.get_column("cvss_source")[0] is None
