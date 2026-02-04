"""
CWEPrimaryTransform - Extract first CWE from cwes array
Tests: Selection logic (first entry), output columns

Layer: Transform/Flatten
Dependencies: transforms.flatten.base
"""

from __future__ import annotations

import numpy as np

from vulnflow.utils import json


def test_cwe_primary_extracts_first():
    """CWEPrimaryTransform extracts first CWE."""
    from vulnflow.dataset import VulnDataset
    from vulnflow.transforms.flatten.cwe import CWEPrimaryTransform

    ds = VulnDataset()
    ds._data = {
        "cve_id": np.array(["CVE-2021-44228"]),
        "cwes": np.array([json.dumps([
            {"id": "CWE-502", "name": "Deserialization of Untrusted Data"},
            {"id": "CWE-20", "name": "Improper Input Validation"},
        ])]),
    }

    CWEPrimaryTransform().transform(ds)

    assert ds.get_column("cwe_id")[0] == "CWE-502"
    assert ds.get_column("cwe_name")[0] == "Deserialization of Untrusted Data"


def test_cwe_primary_handles_empty_array():
    """CWEPrimaryTransform handles empty cwes gracefully."""
    from vulnflow.dataset import VulnDataset
    from vulnflow.transforms.flatten.cwe import CWEPrimaryTransform

    ds = VulnDataset()
    ds._data = {
        "cve_id": np.array(["CVE-2021-44228"]),
        "cwes": np.array(["[]"]),
    }

    CWEPrimaryTransform().transform(ds)

    assert ds.get_column("cwe_id")[0] is None
    assert ds.get_column("cwe_name")[0] is None


def test_cwe_primary_handles_null_name():
    """CWEPrimaryTransform handles CWE entries without name."""
    from vulnflow.dataset import VulnDataset
    from vulnflow.transforms.flatten.cwe import CWEPrimaryTransform

    ds = VulnDataset()
    ds._data = {
        "cve_id": np.array(["CVE-2021-44228"]),
        "cwes": np.array([json.dumps([
            {"id": "CWE-502", "name": None},
        ])]),
    }

    CWEPrimaryTransform().transform(ds)

    assert ds.get_column("cwe_id")[0] == "CWE-502"
    assert ds.get_column("cwe_name")[0] is None
