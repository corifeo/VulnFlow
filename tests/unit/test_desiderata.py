"""
CVD Desiderata - Test desiderata fields computed by CVDStateTransform
Tests: Desiderata mask, count, and anti-desiderata flags

Layer: Transform/Computed
Dependencies: EventInference, CVDStateTransform
"""

from __future__ import annotations

from vulnflow.dataset import VulnDataset
from vulnflow.transforms.computed.cvd_state import CVDStateTransform
from vulnflow.transforms.enrichers.kev import KEVEnricher
from vulnflow.transforms.inferences.events import EventInference


def test_desiderata_mask_and_count():
    """CVDStateTransform creates desiderata_mask and desiderata_count columns."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    # Run dependency chain
    EventInference().transform(dataset)
    CVDStateTransform().transform(dataset)

    # Check desiderata columns exist
    assert "desiderata_mask" in dataset.columns
    assert "desiderata_count" in dataset.columns

    counts = dataset.get_column("desiderata_count")
    # Counts should be integers between 0 and 12
    for count in counts:
        assert 0 <= count <= 12


def test_zero_day_flags():
    """CVDStateTransform creates zero-day boolean flags."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    EventInference().transform(dataset)
    CVDStateTransform().transform(dataset)

    # Check zero-day flags exist
    assert "is_zero_day_exploit" in dataset.columns
    assert "is_zero_day_attack" in dataset.columns
    assert "is_uncoordinated" in dataset.columns


def test_exposure_flags():
    """CVDStateTransform creates exposure boolean flags."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    EventInference().transform(dataset)
    CVDStateTransform().transform(dataset)

    # Check exposure flags exist
    assert "is_premature_disclosure" in dataset.columns
    assert "is_exploit_before_fix" in dataset.columns
    assert "is_attack_before_fix" in dataset.columns


def test_all_19_fields_created():
    """CVDStateTransform creates all 19 CVD analytics fields."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    EventInference().transform(dataset)
    CVDStateTransform().transform(dataset)

    expected_fields = [
        # Core (2)
        "cvd_state",
        "state_bits",
        # Dimensions (2)
        "fix_path",
        "threat_state",
        # Pairwise (1)
        "pair_mask",
        # Desiderata (2)
        "desiderata_mask",
        "desiderata_count",
        # Zero-Day (3)
        "is_zero_day_exploit",
        "is_zero_day_attack",
        "is_uncoordinated",
        # Exposure (3)
        "is_premature_disclosure",
        "is_exploit_before_fix",
        "is_attack_before_fix",
        # Time deltas (6)
        "fix_lag_days",
        "disclosure_lag_days",
        "deployment_lag_days",
        "exploit_lag_days",
        "attack_lag_days",
        "public_to_attack_days",
    ]

    for field in expected_fields:
        assert field in dataset.columns, f"Missing field: {field}"


def test_desiderata_with_kev_enrichment():
    """Desiderata computed correctly with KEV enrichment (attack timestamps)."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    # Enrich with KEV (sets A timestamp)
    dataset.transform([KEVEnricher(source="tests/fixtures/kev_sample.csv")])

    # Run full chain
    EventInference().transform(dataset)
    CVDStateTransform().transform(dataset)

    # CVE-2024-0001: P (from nvd) should be before A (from KEV)
    # NVD published: 2024-01-10, KEV added: 2024-01-11
    # Check that is_attack_before_fix is computed (F doesn't exist, so should be False)
    is_attack_before_fix = dataset.get_column("is_attack_before_fix")
    # Without F timestamp, this should be False (both must exist for True)
    assert not is_attack_before_fix[0]


def test_cvd_state_transform_registered():
    """CVDStateTransform is registered in transform registry."""
    from vulnflow.registry import get_transforms

    transforms = get_transforms()
    transform_names = [t.__name__ for t in transforms]

    assert "CVDStateTransform" in transform_names
