"""
Event Inference - Infer CVD timeline timestamps
Tests: URL date extraction, P/A inference from NVD/KEV

Layer: Transform/Inferences
Dependencies: dataset, KEV enricher
"""

from __future__ import annotations

from vulnflow.dataset import VulnDataset
from vulnflow.transforms.enrichers.kev import KEVEnricher
from vulnflow.transforms.inferences.events import EventInference, extract_date_from_url


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

    # First enrich with KEV
    dataset.transform([KEVEnricher(source="tests/fixtures/kev_sample.csv")])

    # Then run EventInference
    EventInference().transform(dataset)

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
    from vulnflow.registry import get_transforms

    # Import to ensure registration
    from vulnflow.transforms.inferences.events import EventInference  # noqa: F401

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
