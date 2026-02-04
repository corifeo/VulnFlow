"""
CPEPrimaryTransform - Extract first CPE from cpes array
Tests: Selection logic (first entry), output columns

Layer: Transform/Flatten
Dependencies: transforms.flatten.base
"""

from __future__ import annotations

import numpy as np

from vulnflow.utils import json


def test_cpe_primary_extracts_first():
    """CPEPrimaryTransform extracts first CPE by default."""
    from vulnflow.dataset import VulnDataset
    from vulnflow.transforms.flatten.cpe import CPEPrimaryTransform

    ds = VulnDataset()
    ds._data = {
        "cve_id": np.array(["CVE-2021-44228"]),
        "cpes": np.array([json.dumps([
            {"vendor": "apache", "product": "log4j", "version": "2.14.1"},
            {"vendor": "debian", "product": "debian_linux", "version": "10.0"},
        ])]),
    }

    CPEPrimaryTransform().transform(ds)

    assert ds.get_column("cpe_vendor")[0] == "apache"
    assert ds.get_column("cpe_product")[0] == "log4j"
    assert ds.get_column("cpe_version")[0] == "2.14.1"


def test_cpe_primary_handles_empty_array():
    """CPEPrimaryTransform handles empty cpes gracefully."""
    from vulnflow.dataset import VulnDataset
    from vulnflow.transforms.flatten.cpe import CPEPrimaryTransform

    ds = VulnDataset()
    ds._data = {
        "cve_id": np.array(["CVE-2021-44228"]),
        "cpes": np.array(["[]"]),
    }

    CPEPrimaryTransform().transform(ds)

    assert ds.get_column("cpe_vendor")[0] is None
    assert ds.get_column("cpe_product")[0] is None


def test_cpe_primary_extracts_fix_version():
    """CPEPrimaryTransform extracts fix_version if present."""
    from vulnflow.dataset import VulnDataset
    from vulnflow.transforms.flatten.cpe import CPEPrimaryTransform

    ds = VulnDataset()
    ds._data = {
        "cve_id": np.array(["CVE-2021-44228"]),
        "cpes": np.array([json.dumps([
            {
                "vendor": "apache",
                "product": "log4j",
                "version": "2.14.1",
                "fix_version": "2.17.0",
            },
        ])]),
    }

    CPEPrimaryTransform().transform(ds)

    assert ds.get_column("cpe_fix_version")[0] == "2.17.0"
