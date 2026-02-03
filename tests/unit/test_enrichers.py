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
    await enricher.transform(dataset)

    assert "kev_date_added" in dataset.columns
    assert "kev_due_date" in dataset.columns
    assert "kev_required_action" in dataset.columns


@pytest.mark.asyncio
async def test_kev_enricher_matches_by_cve_id():
    """KEVEnricher matches by CVE ID."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    enricher = KEVEnricher(source="tests/fixtures/kev_sample.csv")
    await enricher.transform(dataset)

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
    await enricher.transform(dataset)

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


def test_kev_enricher_vectorized_matches_correctly():
    """KEVEnricher vectorized implementation matches correctly."""
    from datetime import datetime

    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")
    enricher = KEVEnricher(source="tests/fixtures/kev_sample.csv")

    dataset.transform([enricher])

    kev_date_added = dataset.get_column("kev_date_added")

    # First record should have KEV data
    assert kev_date_added[0] is not None

    # Verify it's still a datetime object
    assert isinstance(kev_date_added[0], datetime)


def test_kev_enricher_inherits_from_base():
    """KEVEnricher inherits from BaseEnricher."""
    from vulnflow.transforms.base import BaseEnricher

    assert issubclass(KEVEnricher, BaseEnricher)


def test_kev_enricher_registers_schema():
    """KEVEnricher automatically registers its schema."""
    import importlib

    from vulnflow.registry import FieldDefinition, Registry

    Registry.clear()

    # Re-import to trigger decorator registration after clear
    import vulnflow.transforms.enrichers.kev

    importlib.reload(vulnflow.transforms.enrichers.kev)

    # Instantiate to trigger schema registration via __init__
    _ = KEVEnricher(source="tests/fixtures/kev_sample.csv")

    columns = Registry.get_component_columns("KEVEnricher")
    assert len(columns) == 6
    assert all(isinstance(col, FieldDefinition) for col in columns)


def test_epss_enricher_inherits_from_base():
    """EPSSEnricher inherits from BaseEnricher."""
    from vulnflow.transforms.base import BaseEnricher
    from vulnflow.transforms.enrichers.epss import EPSSEnricher

    assert issubclass(EPSSEnricher, BaseEnricher)


def test_epss_enricher_registers_schema():
    """EPSSEnricher automatically registers its schema."""
    import importlib

    from vulnflow.registry import FieldDefinition, Registry
    from vulnflow.transforms.enrichers.epss import EPSSEnricher

    Registry.clear()

    # Re-import to trigger decorator registration after clear
    import vulnflow.transforms.enrichers.epss

    importlib.reload(vulnflow.transforms.enrichers.epss)

    # Instantiate to trigger schema registration via __init__
    _ = EPSSEnricher(source="tests/fixtures/epss_sample.csv")

    columns = Registry.get_component_columns("EPSSEnricher")
    assert len(columns) == 4  # 4 fields for EPSS
    assert all(isinstance(col, FieldDefinition) for col in columns)
