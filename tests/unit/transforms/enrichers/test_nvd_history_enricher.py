"""NVDHistoryEnricher - CVE change history enricher tests.

All tests mock HTTP to avoid real API calls.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from vulnflow.dataset import VulnDataset


def _make_history_response(
    changes: list[dict],
    total_results: int | None = None,
    start_index: int = 0,
) -> dict:
    """Build a minimal NVD History API response."""
    wrapped = [{"change": c} for c in changes]
    return {
        "totalResults": total_results if total_results is not None else len(changes),
        "startIndex": start_index,
        "resultsPerPage": len(changes),
        "cveChanges": wrapped,
    }


def _make_dataset(cve_ids: list[str]) -> VulnDataset:
    """Create a minimal VulnDataset with cve_id column."""
    ds = VulnDataset()
    ds._data = {"cve_id": np.array(cve_ids, dtype=object)}
    return ds


@pytest.mark.asyncio
@patch("vulnflow.transforms.enrichers.nvd_history.asyncio.sleep", new_callable=AsyncMock)
@patch("vulnflow.transforms.enrichers.nvd_history.HttpFetcher")
async def test_history_enricher_per_cve_mode(MockFetcher, mock_sleep):
    """NVDHistoryEnricher in per_cve mode fetches history per CVE."""
    from vulnflow.transforms.enrichers.nvd_history import NVDHistoryEnricher

    change = {
        "cveId": "CVE-2024-0001",
        "eventName": "CVE Modified",
        "cveChangeId": "abc-123",
        "created": "2024-06-01T10:00:00.000",
        "details": [{"action": "Changed", "type": "Description"}],
    }
    response = _make_history_response([change])

    instance = MockFetcher.return_value
    fetch_mock = AsyncMock(return_value=(json.dumps(response).encode(), "json"))
    instance.fetch = fetch_mock

    enricher = NVDHistoryEnricher(mode="per_cve")
    dataset = _make_dataset(["CVE-2024-0001"])
    await enricher.transform(dataset)

    assert "nvd_changes" in dataset.columns
    changes = json.loads(dataset.get_column("nvd_changes")[0])
    assert len(changes) == 1
    assert changes[0]["eventName"] == "CVE Modified"


@pytest.mark.asyncio
@patch("vulnflow.transforms.enrichers.nvd_history.asyncio.sleep", new_callable=AsyncMock)
@patch("vulnflow.transforms.enrichers.nvd_history.HttpFetcher")
async def test_history_enricher_date_range_mode(MockFetcher, mock_sleep):
    """NVDHistoryEnricher in date_range mode fetches bulk changes."""
    from vulnflow.transforms.enrichers.nvd_history import NVDHistoryEnricher

    changes = [
        {
            "cveId": "CVE-2024-0001",
            "eventName": "CVE Modified",
            "cveChangeId": "abc-123",
            "created": "2024-06-01T10:00:00.000",
            "details": [],
        },
        {
            "cveId": "CVE-2024-9999",  # Not in dataset — should be discarded
            "eventName": "Initial Analysis",
            "cveChangeId": "def-456",
            "created": "2024-06-02T10:00:00.000",
            "details": [],
        },
    ]
    response = _make_history_response(changes)

    instance = MockFetcher.return_value
    fetch_mock = AsyncMock(return_value=(json.dumps(response).encode(), "json"))
    instance.fetch = fetch_mock

    enricher = NVDHistoryEnricher(
        mode="date_range",
        change_start_date="2024-06-01T00:00:00+00:00",
        change_end_date="2024-06-30T00:00:00+00:00",
    )
    dataset = _make_dataset(["CVE-2024-0001"])
    await enricher.transform(dataset)

    assert "nvd_changes" in dataset.columns
    changes_col = json.loads(dataset.get_column("nvd_changes")[0])
    assert len(changes_col) == 1
    assert changes_col[0]["cveId"] == "CVE-2024-0001"


@pytest.mark.asyncio
@patch("vulnflow.transforms.enrichers.nvd_history.asyncio.sleep", new_callable=AsyncMock)
@patch("vulnflow.transforms.enrichers.nvd_history.HttpFetcher")
async def test_history_enricher_no_changes(MockFetcher, mock_sleep):
    """CVEs with no history get empty array."""
    from vulnflow.transforms.enrichers.nvd_history import NVDHistoryEnricher

    response = _make_history_response([])
    instance = MockFetcher.return_value
    fetch_mock = AsyncMock(return_value=(json.dumps(response).encode(), "json"))
    instance.fetch = fetch_mock

    enricher = NVDHistoryEnricher(mode="per_cve")
    dataset = _make_dataset(["CVE-2024-0001"])
    await enricher.transform(dataset)

    changes = json.loads(dataset.get_column("nvd_changes")[0])
    assert changes == []


@pytest.mark.asyncio
@patch("vulnflow.transforms.enrichers.nvd_history.asyncio.sleep", new_callable=AsyncMock)
@patch("vulnflow.transforms.enrichers.nvd_history.HttpFetcher")
async def test_history_enricher_per_cve_failure_continues(MockFetcher, mock_sleep):
    """Per-CVE mode logs warning and continues on failure."""
    from vulnflow.transforms.enrichers.nvd_history import NVDHistoryEnricher
    from vulnflow.utils.http import HttpError

    instance = MockFetcher.return_value
    fetch_mock = AsyncMock(side_effect=HttpError("API error"))
    instance.fetch = fetch_mock

    enricher = NVDHistoryEnricher(mode="per_cve")
    dataset = _make_dataset(["CVE-2024-0001"])
    await enricher.transform(dataset)

    changes = json.loads(dataset.get_column("nvd_changes")[0])
    assert changes == []


def test_history_enricher_date_range_requires_dates():
    """date_range mode without dates raises ValueError."""
    from vulnflow.transforms.enrichers.nvd_history import NVDHistoryEnricher

    with pytest.raises(ValueError, match="change_start_date and change_end_date required"):
        NVDHistoryEnricher(mode="date_range")


def test_history_enricher_output_schema():
    """NVDHistoryEnricher defines nvd_changes in OUTPUT_SCHEMA."""
    from vulnflow.transforms.enrichers.nvd_history import NVDHistoryEnricher

    field_names = [f.name for f in NVDHistoryEnricher.OUTPUT_SCHEMA]
    assert "nvd_changes" in field_names
