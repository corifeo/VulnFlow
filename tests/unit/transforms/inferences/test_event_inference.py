"""
Event Inference - Infer CVD timeline timestamps
Tests: URL date extraction, P/A inference from NVD/KEV, overwrite, tier selection

Layer: Transform/Inferences
Dependencies: dataset, KEV enricher
"""

from __future__ import annotations

from vulnflow.dataset import VulnDataset
from vulnflow.transforms.enrichers.kev import KEVEnricher
from vulnflow.transforms.flatten.kev import KEVFlattenTransform
from vulnflow.transforms.inferences.events import EventInference, extract_date_from_url

from tests.conftest import make_dataset


def test_extract_date_from_url_slash_pattern():
    """Extract date from YYYY/MM/DD pattern."""
    url = "https://example.com/2024/01/15/advisory.html"
    date = extract_date_from_url(url)

    assert date is not None
    assert date.year == 2024
    assert date.month == 1
    assert date.day == 15


def test_extract_date_from_url_iso_pattern():
    """Extract date from YYYY-MM-DD pattern."""
    url = "https://github.com/releases/tag/2024-01-15"
    date = extract_date_from_url(url)

    assert date is not None
    assert date.year == 2024
    assert date.month == 1
    assert date.day == 15


def test_extract_date_from_url_no_pattern():
    """Return None when no date pattern found."""
    url = "https://example.com/vulnerability"
    date = extract_date_from_url(url)

    assert date is None


def test_event_inference_creates_cvd_columns():
    """EventInference creates all 6 CVD timestamp columns."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    EventInference().transform(dataset)

    assert "v_timestamp" in dataset.columns
    assert "f_timestamp" in dataset.columns
    assert "d_timestamp" in dataset.columns
    assert "p_timestamp" in dataset.columns
    assert "x_timestamp" in dataset.columns
    assert "a_timestamp" in dataset.columns


def test_event_inference_p_from_nvd_published():
    """EventInference infers P from nvd_published_at."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    EventInference().transform(dataset)

    p_timestamps = dataset.get_column("p_timestamp")
    # Should have inferred P from nvd_published_at
    assert p_timestamps[0] is not None


def test_event_inference_a_from_kev():
    """EventInference infers A from KEV date_added."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    # First enrich with KEV, then flatten, then infer
    dataset.transform(
        [
            KEVEnricher(source="tests/fixtures/kev_sample.csv"),
            KEVFlattenTransform(),
            EventInference(),
        ]
    )

    a_timestamps = dataset.get_column("a_timestamp")
    # CVE-2024-0001 is in KEV, should have A timestamp
    assert a_timestamps[0] is not None
    # CVE-2024-0002 is NOT in KEV, should not have A timestamp
    assert a_timestamps[1] is None


def test_event_inference_confidence_threshold():
    """EventInference respects confidence threshold."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    # Threshold > 1.0 should skip all inferences
    EventInference(confidence_threshold=1.5).transform(dataset)

    p_timestamps = dataset.get_column("p_timestamp")
    a_timestamps = dataset.get_column("a_timestamp")

    # No timestamps should be inferred with threshold > 1.0
    assert all(p is None for p in p_timestamps)
    assert all(a is None for a in a_timestamps)


def test_event_inference_registered():
    """EventInference is registered in transform registry."""
    import importlib

    from vulnflow.registry import get_transforms

    # Import and reload to ensure registration (handles registry clears from other tests)
    from vulnflow.transforms.inferences import events

    importlib.reload(events)

    transforms = get_transforms()
    transform_names = [t.__name__ for t in transforms]

    assert "EventInference" in transform_names


def test_event_inference_vectorized_works_correctly():
    """EventInference vectorized implementation infers timestamps correctly."""
    from datetime import datetime

    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    # Apply EventInference
    EventInference().transform(dataset)

    p_timestamp = dataset.get_column("p_timestamp")

    # Should have inferred P from nvd_published_at
    assert p_timestamp[0] is not None

    # Verify it's still a datetime
    assert isinstance(p_timestamp[0], datetime)


# ============================================================================
# Overwrite behavior
# ============================================================================


def test_overwrite_preserves_existing():
    """overwrite=False preserves existing timestamps."""
    from datetime import UTC, datetime

    sentinel = datetime(2000, 1, 1, tzinfo=UTC)
    ds = make_dataset({
        "cve_id": ["CVE-2024-0001"],
        "nvd_published_at": [datetime(2024, 1, 15, tzinfo=UTC)],
        "p_timestamp": [sentinel],
        "p_confidence": [0.99],
    })

    EventInference(overwrite=False).transform(ds)

    # Should keep the sentinel, not replace with nvd_published_at
    assert ds.get_column("p_timestamp")[0] == sentinel
    assert ds.get_column("p_confidence")[0] == 0.99


def test_overwrite_replaces():
    """overwrite=True replaces existing timestamps."""
    from datetime import UTC, datetime

    sentinel = datetime(2000, 1, 1, tzinfo=UTC)
    ds = make_dataset({
        "cve_id": ["CVE-2024-0001"],
        "nvd_published_at": [datetime(2024, 1, 15, tzinfo=UTC)],
        "p_timestamp": [sentinel],
        "p_confidence": [0.99],
    })

    EventInference(overwrite=True).transform(ds)

    # Should replace sentinel with nvd_published_at
    p_ts = ds.get_column("p_timestamp")[0]
    assert p_ts != sentinel
    assert p_ts.year == 2024


# ============================================================================
# Explicit tier selection
# ============================================================================


def test_explicit_tier_selection():
    """Explicit tiers=[1] only runs tier 1."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    # Only run tier 1 (authoritative: P from NVD, A from KEV)
    EventInference(tiers=[1]).transform(dataset)

    # P should be set from tier 1
    p_ts = dataset.get_column("p_timestamp")
    assert p_ts[0] is not None

    # V is only set by tiers 2-5, so with only tier 1, most will be None
    # (unless constraints fill them in from other events)


def test_tier_thresholds_is_class_attribute():
    """TIER_THRESHOLDS is defined as a class attribute, not module-level."""
    assert hasattr(EventInference, "TIER_THRESHOLDS")
    assert isinstance(EventInference.TIER_THRESHOLDS, dict)
    assert 1 in EventInference.TIER_THRESHOLDS
    assert EventInference.TIER_THRESHOLDS[1] == 0.95


def test_event_inference_empty_dataset() -> None:
    """EventInference handles 0-row dataset without error."""
    ds = make_dataset({"cve_id": []})
    EventInference().transform(ds)
    for col in ["v_timestamp", "f_timestamp", "d_timestamp",
                 "p_timestamp", "x_timestamp", "a_timestamp"]:
        assert len(ds.get_column(col)) == 0


def test_should_run_respects_threshold() -> None:
    """EventInference uses BaseInference tier gating."""
    inf = EventInference(confidence_threshold=1.5)
    assert inf._should_run(1) is False
