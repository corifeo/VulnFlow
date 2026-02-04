"""
EPSS Enricher - FIRST EPSS probability scores
Tests: CSV loading, CVE ID matching, column addition

Layer: Transform (enricher, async)
Dependencies: dataset
"""

from __future__ import annotations

import pytest

from vulnflow.dataset import VulnDataset
from vulnflow.transforms.enrichers.epss import EPSSEnricher


@pytest.mark.asyncio
async def test_epss_enricher_adds_columns():
    """EPSSEnricher adds epss_probability and epss_percentile columns."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    enricher = EPSSEnricher(source="tests/fixtures/epss_sample.csv")
    await enricher.transform(dataset)

    assert "epss_probability" in dataset.columns
    assert "epss_percentile" in dataset.columns


@pytest.mark.asyncio
async def test_epss_enricher_matches_by_cve_id():
    """EPSS scores correctly matched to CVE IDs."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    enricher = EPSSEnricher(source="tests/fixtures/epss_sample.csv")
    await enricher.transform(dataset)

    probs = dataset.get_column("epss_probability")
    percentiles = dataset.get_column("epss_percentile")

    # CVE-2024-0001 should match
    assert abs(probs[0] - 0.00044) < 0.00001
    assert abs(percentiles[0] - 0.05123) < 0.00001

    # CVE-2024-0002 should match
    assert abs(probs[1] - 0.97480) < 0.00001
    assert abs(percentiles[1] - 0.99990) < 0.00001

    # CVE-2024-0003 should match
    assert abs(probs[2] - 0.00234) < 0.00001
    assert abs(percentiles[2] - 0.34567) < 0.00001


def test_epss_enricher_sync_wrapper():
    """EPSSEnricher works via sync dataset.transform() API."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    dataset.transform([EPSSEnricher(source="tests/fixtures/epss_sample.csv")])

    assert "epss_probability" in dataset.columns
    assert "epss_percentile" in dataset.columns


def test_epss_enricher_registered():
    """EPSSEnricher is registered in transform registry."""
    from vulnflow.registry import get_transforms

    # Import to ensure registration
    from vulnflow.transforms.enrichers.epss import EPSSEnricher  # noqa: F401

    transforms = get_transforms()
    transform_names = [t.__name__ for t in transforms]

    assert "EPSSEnricher" in transform_names


def test_epss_enricher_updates_sources():
    """EPSS enricher adds 'EPSS' to sources list for matched records."""
    import json

    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    dataset.transform([EPSSEnricher(source="tests/fixtures/epss_sample.csv")])

    sources_col = dataset.get_column("sources")
    # All 3 CVEs in fixture should have EPSS source
    sources_0 = json.loads(sources_col[0])
    assert "EPSS" in sources_0, "Matched record should have EPSS source"


@pytest.mark.asyncio
async def test_epss_enricher_adds_date_and_model_columns():
    """EPSSEnricher adds epss_date and epss_model columns."""
    from datetime import datetime

    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    enricher = EPSSEnricher(source="tests/fixtures/epss_sample.csv")
    await enricher.transform(dataset)

    assert "epss_date" in dataset.columns, "Missing epss_date column"
    assert "epss_model" in dataset.columns, "Missing epss_model column"

    # Verify types for matched records
    epss_dates = dataset.get_column("epss_date")
    epss_models = dataset.get_column("epss_model")

    # First record should have EPSS data
    assert epss_dates[0] is not None
    assert isinstance(epss_dates[0], datetime)
    assert epss_models[0] == 3  # Default model version


def test_epss_enricher_vectorized_matches_correctly():
    """EPSSEnricher vectorized implementation matches correctly."""
    import asyncio

    from vulnflow.dataset import VulnDataset
    from vulnflow.transforms.enrichers.epss import EPSSEnricher

    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")
    enricher = EPSSEnricher(source="tests/fixtures/epss_sample.csv")

    asyncio.run(enricher.transform(dataset))

    epss_probability = dataset.get_column("epss_probability")

    # First record should have EPSS data
    assert epss_probability[0] is not None
    assert 0.0 <= epss_probability[0] <= 1.0


# ============================================================================
# Task 9: EPSSEnricher appends to epss_scores array
# ============================================================================


@pytest.mark.asyncio
async def test_epss_enricher_appends_to_array():
    """EPSSEnricher appends to epss_scores array instead of only setting scalar columns."""
    import numpy as np

    from vulnflow.utils import json

    # Create dataset with existing epss_scores array
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    # Set up initial epss_scores array with old data
    n = len(dataset)
    initial_scores = [
        json.dumps([{"probability": 0.5, "percentile": 0.8, "date": "2023-01-01", "model": 2}])
    ] * n
    dataset.set_column("epss_scores", np.array(initial_scores, dtype=object))

    enricher = EPSSEnricher(source="tests/fixtures/epss_sample.csv")
    await enricher.transform(dataset)

    # Verify epss_scores array has been updated (should have 2 entries now)
    epss_scores_raw = dataset.get_column("epss_scores")[0]
    scores = json.loads(epss_scores_raw)

    # Should have original + new entry
    assert len(scores) == 2, f"Expected 2 entries in epss_scores array, got {len(scores)}"

    # New entry should have current EPSS data
    new_entry = scores[-1]  # Most recent should be appended at end
    assert "probability" in new_entry
    assert "percentile" in new_entry
    assert "date" in new_entry
    assert "model" in new_entry


@pytest.mark.asyncio
async def test_epss_enricher_creates_array_if_missing():
    """EPSSEnricher creates epss_scores array if it doesn't exist."""
    from vulnflow.utils import json

    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    # Remove epss_scores if it exists
    if "epss_scores" in dataset.columns:
        del dataset._data["epss_scores"]

    enricher = EPSSEnricher(source="tests/fixtures/epss_sample.csv")
    await enricher.transform(dataset)

    # Should now have epss_scores column
    assert "epss_scores" in dataset.columns

    # First record should have EPSS data in array
    epss_scores_raw = dataset.get_column("epss_scores")[0]
    scores = json.loads(epss_scores_raw)
    assert len(scores) >= 1
    assert scores[0]["probability"] == pytest.approx(0.00044, abs=0.00001)


@pytest.mark.asyncio
async def test_epss_enricher_deduplicates_by_date():
    """EPSSEnricher doesn't add duplicate entries for same date."""
    import numpy as np

    from vulnflow.utils import json

    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    # Set up initial epss_scores array with SAME date as fixture (2024-01-15)
    n = len(dataset)
    initial_scores = [
        json.dumps([{"probability": 0.1, "percentile": 0.2, "date": "2024-01-15T00:00:00+00:00", "model": 3}])
    ] * n
    dataset.set_column("epss_scores", np.array(initial_scores, dtype=object))

    enricher = EPSSEnricher(source="tests/fixtures/epss_sample.csv")
    await enricher.transform(dataset)

    # Should still have only 1 entry (deduplicated by date)
    epss_scores_raw = dataset.get_column("epss_scores")[0]
    scores = json.loads(epss_scores_raw)

    assert len(scores) == 1, f"Expected 1 entry (deduplicated), got {len(scores)}"
    # The newer value should win
    assert scores[0]["probability"] == pytest.approx(0.00044, abs=0.00001)
