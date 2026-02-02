"""
Enrichers - KEVEnricher and async enrichment
Tests: KEVEnricher CSV parsing, CVE ID matching

Layer: Transform (ETL) - Enrichment
Dependencies: dataset, schema
"""

from __future__ import annotations

import pytest

from vulnflow.dataset import VulnDataset
from vulnflow.transforms.enrichers.kev import KEVEnricher


@pytest.mark.asyncio
async def test_kev_enricher_adds_columns():
    """KEVEnricher adds kev_* columns."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    enricher = KEVEnricher(source="tests/fixtures/kev_sample.csv")
    await enricher.enrich(dataset)

    assert "kev_date_added" in dataset.columns
    assert "kev_due_date" in dataset.columns
    assert "kev_required_action" in dataset.columns


@pytest.mark.asyncio
async def test_kev_enricher_matches_by_cve_id():
    """KEVEnricher matches by CVE ID."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    enricher = KEVEnricher(source="tests/fixtures/kev_sample.csv")
    await enricher.enrich(dataset)

    # Check records based on fixture: CVE-2024-0001 and CVE-2024-0003 in KEV
    kev_dates = dataset.get_column("kev_date_added")
    assert kev_dates[0] is not None  # CVE-2024-0001 in KEV
    assert kev_dates[1] is None  # CVE-2024-0002 NOT in KEV
    assert kev_dates[2] is not None  # CVE-2024-0003 in KEV


@pytest.mark.asyncio
async def test_kev_enricher_parses_dates():
    """KEVEnricher parses date strings to datetime."""
    from datetime import datetime

    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    enricher = KEVEnricher(source="tests/fixtures/kev_sample.csv")
    await enricher.enrich(dataset)

    kev_dates = dataset.get_column("kev_date_added")
    first_date = kev_dates[0]

    assert isinstance(first_date, datetime)
    assert first_date.year == 2024
    assert first_date.month == 1
    assert first_date.day == 11  # From fixture


def test_kev_enricher_sync_wrapper():
    """Sync transform() method works with enrichers via VulnDataset."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    enricher = KEVEnricher(source="tests/fixtures/kev_sample.csv")
    dataset.transform([enricher])

    assert "kev_date_added" in dataset.columns
    assert dataset.get_column("kev_date_added")[0] is not None


def test_dataset_transform_with_empty_list():
    """VulnDataset.transform() with empty list does nothing."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")
    initial_columns = dataset.columns.copy()

    dataset.transform([])

    assert dataset.columns == initial_columns


def test_kev_enricher_registered():
    """KEVEnricher is registered in enricher registry."""
    from vulnflow.registry import get_enrichers

    # Import to ensure registration
    from vulnflow.transforms.enrichers.kev import KEVEnricher  # noqa: F401

    enrichers = get_enrichers()
    enricher_names = [e.__name__ for e in enrichers]

    assert "KEVEnricher" in enricher_names


def test_kev_enricher_updates_sources():
    """KEV enricher adds 'KEV' to sources list for matched records."""
    import json

    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    dataset.transform([KEVEnricher(source="tests/fixtures/kev_sample.csv")])

    sources_col = dataset.get_column("sources")
    # CVE-2024-0001 is in KEV fixture, should have KEV source
    sources_0 = json.loads(sources_col[0])
    assert "KEV" in sources_0, "Matched record should have KEV source"

    # CVE-2024-0002 is NOT in KEV fixture
    sources_1 = json.loads(sources_col[1])
    assert "KEV" not in sources_1, "Unmatched record should not have KEV source"
