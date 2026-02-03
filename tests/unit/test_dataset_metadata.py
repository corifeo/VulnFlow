"""
Dataset Metadata - Sources and timestamp tracking
Tests: update_sources, timestamp updates

Layer: Core dataset operations
Dependencies: None
"""

from __future__ import annotations

import json
from datetime import datetime

from tests.conftest import dataset_from_records

from vulnflow.schema import VulnRecord


def test_update_sources_adds_new_source():
    """update_sources adds source to list without duplicates."""
    records = [VulnRecord(cve_id="CVE-2024-0001", sources=["NVD"])]
    dataset = dataset_from_records(records)

    dataset.update_sources([0], "KEV")

    sources_col = dataset.get_column("sources")
    sources = json.loads(sources_col[0])
    assert "KEV" in sources
    assert "NVD" in sources
    assert len(sources) == 2


def test_update_sources_no_duplicates():
    """update_sources doesn't add duplicate sources."""
    records = [VulnRecord(cve_id="CVE-2024-0001", sources=["NVD", "KEV"])]
    dataset = dataset_from_records(records)

    dataset.update_sources([0], "KEV")

    sources_col = dataset.get_column("sources")
    sources = json.loads(sources_col[0])
    kev_count = sources.count("KEV")
    assert kev_count == 1, "KEV should appear only once"


def test_update_sources_updates_timestamp():
    """update_sources updates last_updated_at timestamp."""
    records = [VulnRecord(cve_id="CVE-2024-0001", sources=["NVD"])]
    dataset = dataset_from_records(records)

    before = datetime.now()
    dataset.update_sources([0], "KEV")
    after = datetime.now()

    updated_at = dataset.get_column("last_updated_at")
    assert updated_at[0] is not None
    assert before <= updated_at[0] <= after
