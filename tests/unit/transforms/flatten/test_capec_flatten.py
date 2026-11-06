"""Tests for CAPECFlattenTransform."""

from __future__ import annotations

import json

import numpy as np

from vulnflow.dataset import VulnDataset
from vulnflow.transforms.flatten.capec import CAPECFlattenTransform


def test_capec_flatten_extracts_primary() -> None:
    """Test CAPECFlattenTransform extracts first CAPEC entry to 4 scalar columns."""
    dataset = VulnDataset()
    dataset._data = {
        "cve_id": np.array(["CVE-2024-0001"], dtype=object),
        "capecs": np.array(
            [
                json.dumps(
                    [
                        {
                            "id": "CAPEC-86",
                            "name": "XSS via HTTP Headers",
                            "abstraction": "Detailed",
                            "status": "Draft",
                            "likelihood_of_attack": "Medium",
                            "typical_severity": "High",
                            "skills_required": ["Medium"],
                            "consequences": [
                                {
                                    "scopes": ["Confidentiality"],
                                    "impacts": ["Read Data"],
                                },
                            ],
                            "attack_techniques": ["1189"],
                        },
                        {
                            "id": "CAPEC-198",
                            "name": "XSS Targeting Error Pages",
                            "abstraction": "Detailed",
                            "status": "Draft",
                            "likelihood_of_attack": "High",
                            "typical_severity": "Medium",
                            "skills_required": None,
                            "consequences": None,
                            "attack_techniques": None,
                        },
                    ]
                )
            ],
            dtype=object,
        ),
    }

    transform = CAPECFlattenTransform()
    transform.transform(dataset)

    assert dataset.get_column("capec_id")[0] == "CAPEC-86"
    assert dataset.get_column("capec_name")[0] == "XSS via HTTP Headers"
    assert dataset.get_column("capec_likelihood")[0] == "Medium"
    assert dataset.get_column("capec_severity")[0] == "High"


def test_capec_flatten_handles_empty() -> None:
    """Test CAPECFlattenTransform handles empty capecs array."""
    dataset = VulnDataset()
    dataset._data = {
        "cve_id": np.array(["CVE-9999-9999"], dtype=object),
        "capecs": np.array(["[]"], dtype=object),
    }

    transform = CAPECFlattenTransform()
    transform.transform(dataset)

    assert dataset.get_column("capec_id")[0] is None
    assert dataset.get_column("capec_name")[0] is None
    assert dataset.get_column("capec_likelihood")[0] is None
    assert dataset.get_column("capec_severity")[0] is None
