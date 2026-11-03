"""Tests for KEVFlattenTransform."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np

from vulnflow.dataset import VulnDataset
from vulnflow.transforms.flatten.kev import KEVFlattenTransform


def test_kev_flatten_extracts_all_fields() -> None:
    """Test KEVFlattenTransform extracts all KEV fields to scalars."""
    dataset = VulnDataset()
    dataset._data = {
        "cve_id": np.array(["CVE-2024-0001"], dtype=object),
        "kev": np.array(
            [
                json.dumps(
                    {
                        "date_added": "2024-01-01T00:00:00Z",
                        "due_date": "2024-02-01T00:00:00Z",
                        "vendor": "Vendor",
                        "product": "Product",
                        "vulnerability_name": "Test Vuln",
                        "required_action": "Apply patch",
                        "known_ransomware_use": True,
                    }
                )
            ],
            dtype=object,
        ),
    }

    transform = KEVFlattenTransform()
    transform.transform(dataset)

    assert dataset.get_column("is_kev")[0] is True
    assert dataset.get_column("kev_date_added")[0] == datetime(2024, 1, 1, tzinfo=UTC)
    assert dataset.get_column("kev_due_date")[0] == datetime(2024, 2, 1, tzinfo=UTC)
    assert dataset.get_column("kev_vendor")[0] == "Vendor"
    assert dataset.get_column("kev_product")[0] == "Product"
    assert dataset.get_column("kev_vulnerability_name")[0] == "Test Vuln"
    assert dataset.get_column("kev_required_action")[0] == "Apply patch"
    assert dataset.get_column("kev_ransomware")[0] is True


def test_kev_flatten_handles_null() -> None:
    """Test KEVFlattenTransform handles null KEV entries."""
    dataset = VulnDataset()
    dataset._data = {
        "cve_id": np.array(["CVE-9999-9999"], dtype=object),
        "kev": np.array(["null"], dtype=object),
    }

    transform = KEVFlattenTransform()
    transform.transform(dataset)

    assert dataset.get_column("is_kev")[0] is False
    assert dataset.get_column("kev_date_added")[0] is None
    assert dataset.get_column("kev_vendor")[0] is None
