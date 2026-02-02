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
    """EPSSEnricher is registered in enricher registry."""
    from vulnflow.registry import get_enrichers

    # Import to ensure registration
    from vulnflow.transforms.enrichers.epss import EPSSEnricher  # noqa: F401

    enrichers = get_enrichers()
    enricher_names = [e.__name__ for e in enrichers]

    assert "EPSSEnricher" in enricher_names


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
