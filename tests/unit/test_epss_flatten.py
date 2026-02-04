"""
EPSSFlattenTransform - Select most recent EPSS score
Tests: Selection logic (max date), output columns

Layer: Transform/Flatten
Dependencies: transforms.flatten.base
"""

from __future__ import annotations

import numpy as np

from vulnflow.utils import json


def test_epss_flatten_selects_most_recent():
    """EPSSFlattenTransform selects entry with latest date."""
    from vulnflow.dataset import VulnDataset
    from vulnflow.transforms.flatten.epss import EPSSFlattenTransform

    ds = VulnDataset()
    ds._data = {
        "cve_id": np.array(["CVE-2021-44228"]),
        "epss_scores": np.array([json.dumps([
            {"probability": 0.5, "percentile": 0.8, "date": "2024-01-01T00:00:00+00:00", "model": 3},
            {"probability": 0.9, "percentile": 0.95, "date": "2024-02-01T00:00:00+00:00", "model": 3},
        ])]),
    }

    EPSSFlattenTransform().transform(ds)

    assert ds.get_column("epss_probability")[0] == 0.9
    assert ds.get_column("epss_percentile")[0] == 0.95


def test_epss_flatten_handles_empty_array():
    """EPSSFlattenTransform handles empty epss_scores gracefully."""
    from vulnflow.dataset import VulnDataset
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
    from vulnflow.dataset import VulnDataset
    from vulnflow.transforms.flatten.epss import EPSSFlattenTransform

    ds = VulnDataset()
    ds._data = {
        "cve_id": np.array(["CVE-2021-44228"]),
        "epss_scores": np.array([json.dumps([
            {"probability": 0.9, "percentile": 0.95, "date": "2024-02-01T00:00:00+00:00", "model": 3},
        ])]),
    }

    EPSSFlattenTransform().transform(ds)

    assert "epss_probability" in ds.columns
    assert "epss_percentile" in ds.columns
    assert "epss_date" in ds.columns
    assert "epss_model" in ds.columns
