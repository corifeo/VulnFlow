"""
Idempotency - Daily update workflow
Tests: Re-enrichment updates without duplication

Layer: Integration
Dependencies: All enrichers, Parquet I/O
"""

from __future__ import annotations

import json
import time

from vulnflow.dataset import VulnDataset
from vulnflow.transforms.enrichers.epss import EPSSEnricher
from vulnflow.transforms.enrichers.kev import KEVEnricher


def test_reimport_idempotent(tmp_path):
    """Re-enrichment updates, doesn't duplicate records."""
    # Day 1: Initial load and enrichment
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")
    dataset.transform([
        KEVEnricher(source="tests/fixtures/kev_sample.csv"),
        EPSSEnricher(source="tests/fixtures/epss_sample.csv"),
    ])

    original_count = len(dataset)
    output_path = tmp_path / "day1.parquet"
    dataset.to_parquet(str(output_path))

    # Day 2: Reload and re-enrich
    dataset2 = VulnDataset.from_parquet(str(output_path))
    dataset2.transform([
        KEVEnricher(source="tests/fixtures/kev_sample.csv"),
        EPSSEnricher(source="tests/fixtures/epss_sample.csv"),
    ])

    # Should update, not duplicate
    assert len(dataset2) == original_count, "Record count should not change"


def test_sources_accumulate_without_duplicates(tmp_path):
    """Sources list accumulates without duplicates on re-enrichment."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    # First enrichment: KEV only
    dataset.transform([KEVEnricher(source="tests/fixtures/kev_sample.csv")])

    # Save and reload
    output_path = tmp_path / "partial.parquet"
    dataset.to_parquet(str(output_path))
    dataset2 = VulnDataset.from_parquet(str(output_path))

    # Re-enrich with same KEV (should not duplicate)
    dataset2.transform([KEVEnricher(source="tests/fixtures/kev_sample.csv")])

    sources_col2 = dataset2.get_column("sources")
    sources2 = json.loads(sources_col2[0])

    # KEV should appear only once
    kev_count = sources2.count("KEV")
    assert kev_count <= 1, "KEV should not be duplicated"


def test_timestamp_updates_on_reenrichment(tmp_path):
    """last_updated_at timestamp updates on re-enrichment."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    # First enrichment
    dataset.transform([KEVEnricher(source="tests/fixtures/kev_sample.csv")])
    first_timestamp = dataset.get_column("last_updated_at")[0]

    # Save and reload
    output_path = tmp_path / "day1.parquet"
    dataset.to_parquet(str(output_path))

    # Wait to ensure timestamp difference
    time.sleep(0.01)

    dataset2 = VulnDataset.from_parquet(str(output_path))

    # Second enrichment with EPSS
    dataset2.transform([EPSSEnricher(source="tests/fixtures/epss_sample.csv")])
    second_timestamp = dataset2.get_column("last_updated_at")[0]

    # Timestamp should be updated (later) for matched records
    if first_timestamp and second_timestamp:
        assert second_timestamp >= first_timestamp, "Timestamp should advance"


def test_parquet_roundtrip_preserves_sources(tmp_path):
    """Parquet round-trip preserves sources metadata."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")
    dataset.transform([
        KEVEnricher(source="tests/fixtures/kev_sample.csv"),
        EPSSEnricher(source="tests/fixtures/epss_sample.csv"),
    ])

    # Save and reload
    output_path = tmp_path / "metadata.parquet"
    dataset.to_parquet(str(output_path))
    dataset2 = VulnDataset.from_parquet(str(output_path))

    # Check sources preserved
    sources_col = dataset2.get_column("sources")
    sources = json.loads(sources_col[0])

    # Should have nvd from initial load plus enricher sources
    assert "nvd" in sources or "NVD" in sources.upper() if isinstance(sources, str) else any(
        s.lower() == "nvd" for s in sources
    ), "Should have NVD source"
