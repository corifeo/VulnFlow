"""Tests for STRIDEFlattenTransform - Extract STRIDE threat booleans.

Layer: Tests/Unit
Dependencies: transforms.flatten.stride
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from vulnflow.dataset import VulnDataset
from vulnflow.transforms.flatten.stride import STRIDEFlattenTransform

# ============================================================================
# Helpers
# ============================================================================


def _make_dataset(columns: dict[str, list[Any]]) -> VulnDataset:
    """Create a VulnDataset from column lists."""
    dataset = VulnDataset()
    dataset._data = {k: np.array(v, dtype=object) for k, v in columns.items()}
    return dataset


# ============================================================================
# Tests
# ============================================================================


def test_stride_flatten_sets_booleans() -> None:
    """stride_threats array → 6 boolean columns set correctly."""
    threats = json.dumps([
        {"category": "spoofing", "confidence": 0.95, "source": "capec_consequences", "capec_ids": []},
        {"category": "tampering", "confidence": 0.85, "source": "cwe_mapping", "capec_ids": []},
        {"category": "dos", "confidence": 0.75, "source": "cvss_impact", "capec_ids": []},
    ])
    ds = _make_dataset({
        "cve_id": ["CVE-2024-0001"],
        "stride_threats": [threats],
    })
    STRIDEFlattenTransform().transform(ds)

    assert ds.get_column("stride_spoofing")[0] is True
    assert ds.get_column("stride_tampering")[0] is True
    assert ds.get_column("stride_dos")[0] is True
    # Not assigned
    assert ds.get_column("stride_repudiation")[0] is False
    assert ds.get_column("stride_info_disclosure")[0] is False
    assert ds.get_column("stride_elevation")[0] is False


def test_stride_flatten_empty_defaults_false() -> None:
    """Empty threats array → all 6 booleans are False (not None)."""
    ds = _make_dataset({
        "cve_id": ["CVE-2024-0001"],
        "stride_threats": ["[]"],
    })
    STRIDEFlattenTransform().transform(ds)

    assert ds.get_column("stride_spoofing")[0] is False
    assert ds.get_column("stride_tampering")[0] is False
    assert ds.get_column("stride_repudiation")[0] is False
    assert ds.get_column("stride_info_disclosure")[0] is False
    assert ds.get_column("stride_dos")[0] is False
    assert ds.get_column("stride_elevation")[0] is False
