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
    await enricher.enrich(dataset)

    assert "epss_probability" in dataset.columns
    assert "epss_percentile" in dataset.columns


@pytest.mark.asyncio
async def test_epss_enricher_matches_by_cve_id():
    """EPSS scores correctly matched to CVE IDs."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    enricher = EPSSEnricher(source="tests/fixtures/epss_sample.csv")
    await enricher.enrich(dataset)

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
