"""
CVD Desiderata Transform - Compute 12 desiderata flags
Tests: Flag computation, dependency chain

Layer: Transform/Computed
Dependencies: EventInference, CVDStateTransform
"""

from __future__ import annotations

from vulnflow.dataset import VulnDataset
from vulnflow.transforms.computed.cvd_state import CVDStateTransform
from vulnflow.transforms.computed.desiderata import CVDDesiderataTransform
from vulnflow.transforms.enrichers.kev import KEVEnricher
from vulnflow.transforms.inferences.events import EventInference


def test_desiderata_creates_columns():
    """CVDDesiderataTransform creates desiderata_d1 through desiderata_d12."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    # Run dependency chain
    EventInference().transform(dataset)
    CVDStateTransform().transform(dataset)
    CVDDesiderataTransform().transform(dataset)

    # Check all 12 desiderata columns exist
    for i in range(1, 13):
        assert f"desiderata_d{i}" in dataset.columns


def test_desiderata_count_column():
    """CVDDesiderataTransform creates desiderata_count column."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    EventInference().transform(dataset)
    CVDStateTransform().transform(dataset)
    CVDDesiderataTransform().transform(dataset)

    assert "desiderata_count" in dataset.columns
    counts = dataset.get_column("desiderata_count")
    # Counts should be integers between 0 and 12
    for count in counts:
        assert 0 <= count <= 12


def test_desiderata_d11_p_before_a():
    """D11 is True when P is before A."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    # Enrich with KEV (sets A timestamp)
    dataset.transform([KEVEnricher(source="tests/fixtures/kev_sample.csv")])

    # Run full chain
    EventInference().transform(dataset)
    CVDStateTransform().transform(dataset)
    CVDDesiderataTransform().transform(dataset)

    d11 = dataset.get_column("desiderata_d11")
    # CVE-2024-0001: P (from nvd) should be before A (from KEV)
    # NVD published: 2024-01-10, KEV added: 2024-01-11
    assert d11[0] is True  # P before A


def test_desiderata_registered():
    """CVDDesiderataTransform is registered in transform registry."""
    from vulnflow.registry import get_transforms

    # Import to ensure registration
    from vulnflow.transforms.computed.desiderata import (
        CVDDesiderataTransform,  # noqa: F401
    )

    transforms = get_transforms()
    transform_names = [t.__name__ for t in transforms]

    assert "CVDDesiderataTransform" in transform_names
