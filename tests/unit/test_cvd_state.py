"""
CVD State Transform - Compute state from timestamps
Tests: State string computation, dependency on EventInference

Layer: Transform/Computed
Dependencies: EventInference
"""

from __future__ import annotations

from vulnflow.dataset import VulnDataset
from vulnflow.transforms.computed.cvd_state import CVDStateTransform
from vulnflow.transforms.enrichers.kev import KEVEnricher
from vulnflow.transforms.inferences.events import EventInference


def test_cvd_state_creates_state_column():
    """CVDStateTransform creates cvd_state column."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    # Run EventInference first (provides CVD timestamps)
    EventInference().transform(dataset)

    # Then run CVDStateTransform
    CVDStateTransform().transform(dataset)

    assert "cvd_state" in dataset.columns


def test_cvd_state_format():
    """CVD state is 6 characters (VFdPxa format)."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    EventInference().transform(dataset)
    CVDStateTransform().transform(dataset)

    states = dataset.get_column("cvd_state")
    for state in states:
        assert len(state) == 6
        # Each position should be uppercase (event occurred) or lowercase (not occurred)
        assert state[0] in "Vv"
        assert state[1] in "Ff"
        assert state[2] in "Dd"
        assert state[3] in "Pp"
        assert state[4] in "Xx"
        assert state[5] in "Aa"


def test_cvd_state_reflects_timestamps():
    """CVD state reflects which timestamps are set."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    # EventInference sets P from nvd_published_at
    EventInference().transform(dataset)
    CVDStateTransform().transform(dataset)

    states = dataset.get_column("cvd_state")
    # First record should have P (public disclosure) from nvd_published_at
    assert states[0][3] == "P"  # P is at position 3


def test_cvd_state_with_kev():
    """CVD state reflects A timestamp from KEV."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    # Enrich with KEV first
    dataset.transform([KEVEnricher(source="tests/fixtures/kev_sample.csv")])

    # Then run EventInference and CVDStateTransform
    EventInference().transform(dataset)
    CVDStateTransform().transform(dataset)

    states = dataset.get_column("cvd_state")
    # CVE-2024-0001 is in KEV, should have A
    assert states[0][5] == "A"  # A is at position 5
    # CVE-2024-0002 is NOT in KEV, should not have A
    assert states[1][5] == "a"


def test_cvd_state_registered():
    """CVDStateTransform is registered in transform registry."""
    from vulnflow.registry import get_transforms

    # Import to ensure registration
    from vulnflow.transforms.computed.cvd_state import CVDStateTransform  # noqa: F401

    transforms = get_transforms()
    transform_names = [t.__name__ for t in transforms]

    assert "CVDStateTransform" in transform_names
