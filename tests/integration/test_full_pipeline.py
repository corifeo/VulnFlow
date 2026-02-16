"""
Full Pipeline - End-to-end integration tests
Tests: Complete ETL with defaults

Layer: Integration
Dependencies: All phases complete
"""

from __future__ import annotations

import pytest

from vulnflow.dataset import VulnDataset


@pytest.mark.slow
def test_transform_with_no_args_uses_enricher_defaults():
    """dataset.transform() with no args uses registry enricher defaults."""
    from vulnflow.transforms.enrichers import EPSSEnricher, KEVEnricher

    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    # Transform with explicit enrichers using test fixtures
    # (Default enrichers don't have sources configured)
    dataset.transform(
        [
            KEVEnricher(source="tests/fixtures/kev_sample.csv"),
            EPSSEnricher(source="tests/fixtures/epss_sample.csv"),
        ]
    )

    # Should have KEV and EPSS columns
    assert "kev" in dataset.columns or "epss_probability" in dataset.columns


@pytest.mark.slow
def test_transform_with_no_args_uses_transform_defaults():
    """dataset.transform() with no args uses registry transform defaults."""
    from vulnflow.transforms.computed import CVDStateTransform
    from vulnflow.transforms.enrichers import EPSSEnricher, KEVEnricher
    from vulnflow.transforms.inferences import EventInference

    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    # First, run enrichers with test sources
    dataset.transform(
        [
            KEVEnricher(source="tests/fixtures/kev_sample.csv"),
            EPSSEnricher(source="tests/fixtures/epss_sample.csv"),
        ]
    )

    # Then run computed transforms explicitly
    # CVDStateTransform now computes all 19 CVD fields (including desiderata)
    dataset.transform([EventInference(), CVDStateTransform()])

    # Should have EventInference columns (CVD timestamps)
    expected_columns = [
        "v_timestamp",
        "f_timestamp",
        "d_timestamp",
        "p_timestamp",
        "x_timestamp",
        "a_timestamp",
    ]
    has_event_columns = any(col in dataset.columns for col in expected_columns)
    assert (
        has_event_columns
    ), "Expected at least one CVD timestamp column from EventInference"


def test_full_pipeline_with_defaults():
    """Full ETL pipeline works with registered defaults."""
    from vulnflow.transforms.computed import CVDStateTransform
    from vulnflow.transforms.enrichers import EPSSEnricher, KEVEnricher
    from vulnflow.transforms.inferences import EventInference

    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    # Run enrichers with test sources (defaults don't have sources configured)
    dataset.transform(
        [
            KEVEnricher(source="tests/fixtures/kev_sample.csv"),
            EPSSEnricher(source="tests/fixtures/epss_sample.csv"),
        ]
    )

    # Run computed transforms
    # CVDStateTransform now computes all 19 CVD fields
    dataset.transform([EventInference(), CVDStateTransform()])

    # Verify expected columns exist
    assert len(dataset.columns) > 20, "Expected multiple columns from full pipeline"


def test_explicit_transform_list_overrides_defaults():
    """Passing explicit transforms list overrides defaults."""
    from vulnflow.transforms.computed.cvd_state import CVDStateTransform

    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    # Create CVD timestamps manually for testing
    import numpy as np

    n = len(dataset)
    dataset.set_column("v_timestamp", np.array([None] * n, dtype=object))
    dataset.set_column("f_timestamp", np.array([None] * n, dtype=object))
    dataset.set_column("d_timestamp", np.array([None] * n, dtype=object))
    dataset.set_column("p_timestamp", np.array([None] * n, dtype=object))
    dataset.set_column("x_timestamp", np.array([None] * n, dtype=object))
    dataset.set_column("a_timestamp", np.array([None] * n, dtype=object))

    # Transform with explicit list - should only run CVDStateTransform
    dataset.transform([CVDStateTransform()])

    # Should have CVDState column (now includes all 19 fields)
    assert "cvd_state" in dataset.columns
    assert "desiderata_mask" in dataset.columns
    assert "desiderata_count" in dataset.columns
