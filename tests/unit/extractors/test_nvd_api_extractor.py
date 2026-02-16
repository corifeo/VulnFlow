"""NVDAPIExtractor - query-oriented NVD API extractor tests.

All tests mock HTTP to avoid real API calls.
"""

from __future__ import annotations

import importlib
import json
from unittest.mock import patch

import pytest

from vulnflow.dataset import VulnDataset


def _make_api_response(
    cve_ids: list[str],
    total_results: int | None = None,
    start_index: int = 0,
) -> dict:
    """Build a minimal NVD API response for testing."""
    vulns = []
    for cve_id in cve_ids:
        vulns.append(
            {
                "cve": {
                    "id": cve_id,
                    "descriptions": [{"lang": "en", "value": f"Test {cve_id}"}],
                    "published": "2024-01-10T10:00:00.000Z",
                    "lastModified": "2024-01-12T15:30:00.000Z",
                    "vulnStatus": "Analyzed",
                    "metrics": {},
                    "weaknesses": [],
                    "configurations": [],
                    "references": [],
                }
            }
        )
    return {
        "totalResults": total_results if total_results is not None else len(cve_ids),
        "startIndex": start_index,
        "resultsPerPage": len(cve_ids),
        "format": "NVD_CVE",
        "version": "2.0",
        "vulnerabilities": vulns,
    }


@patch("vulnflow.extractors.nvd_api.HttpFetcher")
def test_nvd_api_extractor_single_cve(MockFetcher):
    """NVDAPIExtractor fetches a single CVE by ID."""
    from vulnflow.extractors.nvd_api import NVDAPIExtractor

    response = _make_api_response(["CVE-2024-3094"])
    instance = MockFetcher.return_value
    instance.fetch_sync.return_value = (json.dumps(response).encode(), "json")

    extractor = NVDAPIExtractor(cve_id="CVE-2024-3094")
    dataset = extractor.extract()

    assert isinstance(dataset, VulnDataset)
    assert len(dataset) == 1
    assert dataset.get_column("cve_id")[0] == "CVE-2024-3094"


@patch("vulnflow.extractors.nvd_api.time.sleep")
@patch("vulnflow.extractors.nvd_api.HttpFetcher")
def test_nvd_api_extractor_pagination(MockFetcher, _mock_sleep):
    """NVDAPIExtractor paginates through multiple pages."""
    from vulnflow.extractors.nvd_api import NVDAPIExtractor

    page1 = _make_api_response(
        ["CVE-2024-0001", "CVE-2024-0002"], total_results=3, start_index=0
    )
    page2 = _make_api_response(["CVE-2024-0003"], total_results=3, start_index=2)

    instance = MockFetcher.return_value
    instance.fetch_sync.side_effect = [
        (json.dumps(page1).encode(), "json"),
        (json.dumps(page2).encode(), "json"),
    ]

    extractor = NVDAPIExtractor(keyword="test", results_per_page=2)
    dataset = extractor.extract()

    assert len(dataset) == 3
    cve_ids = list(dataset.get_column("cve_id"))
    assert cve_ids == ["CVE-2024-0001", "CVE-2024-0002", "CVE-2024-0003"]


@patch("vulnflow.extractors.nvd_api.HttpFetcher")
def test_nvd_api_extractor_empty_results(MockFetcher):
    """NVDAPIExtractor returns empty dataset when no results."""
    from vulnflow.extractors.nvd_api import NVDAPIExtractor

    response = _make_api_response([], total_results=0)
    instance = MockFetcher.return_value
    instance.fetch_sync.return_value = (json.dumps(response).encode(), "json")

    extractor = NVDAPIExtractor(keyword="nonexistent")
    dataset = extractor.extract()

    assert isinstance(dataset, VulnDataset)
    assert len(dataset) == 0


@patch("vulnflow.extractors.nvd_api.time.sleep")
@patch("vulnflow.extractors.nvd_api.HttpFetcher")
def test_nvd_api_extractor_date_chunking(MockFetcher, _mock_sleep):
    """NVDAPIExtractor auto-chunks date ranges >120 days.

    A 200-day range (2024-01-01 to 2024-07-19) produces exactly 2 chunks:
    chunk 1: Jan 1 - Apr 30, chunk 2: Apr 30 - Jul 19.
    """
    from vulnflow.extractors.nvd_api import NVDAPIExtractor

    chunk1 = _make_api_response(["CVE-2024-0001"])
    chunk2 = _make_api_response(["CVE-2024-0002"])

    instance = MockFetcher.return_value
    instance.fetch_sync.side_effect = [
        (json.dumps(chunk1).encode(), "json"),
        (json.dumps(chunk2).encode(), "json"),
    ]

    # 200 days: exactly 2 chunks of 120 days
    extractor = NVDAPIExtractor(
        pub_start_date="2024-01-01T00:00:00+00:00",
        pub_end_date="2024-07-19T00:00:00+00:00",
    )
    dataset = extractor.extract()

    assert len(dataset) == 2
    assert instance.fetch_sync.call_count == 2


@patch("vulnflow.extractors.nvd_api.HttpFetcher")
def test_nvd_api_extractor_api_key_header(MockFetcher):
    """NVDAPIExtractor passes API key in request."""
    from vulnflow.extractors.nvd_api import NVDAPIExtractor

    response = _make_api_response(["CVE-2024-0001"])
    instance = MockFetcher.return_value
    instance.fetch_sync.return_value = (json.dumps(response).encode(), "json")

    extractor = NVDAPIExtractor(cve_id="CVE-2024-0001", api_key="test-key")
    extractor.extract()

    assert extractor._api_key == "test-key"


@patch("vulnflow.extractors.nvd_api.time.sleep")
@patch("vulnflow.extractors.nvd_api.HttpFetcher")
def test_nvd_api_extractor_deduplicates_cves(MockFetcher, _mock_sleep):
    """NVDAPIExtractor deduplicates CVEs from overlapping chunks.

    Uses a 200-day range producing 2 chunks. Both chunks return
    the same CVE ID; the result should contain only 1 record.
    """
    from vulnflow.extractors.nvd_api import NVDAPIExtractor

    chunk1 = _make_api_response(["CVE-2024-0001"])
    chunk2 = _make_api_response(["CVE-2024-0001"])

    instance = MockFetcher.return_value
    instance.fetch_sync.side_effect = [
        (json.dumps(chunk1).encode(), "json"),
        (json.dumps(chunk2).encode(), "json"),
    ]

    extractor = NVDAPIExtractor(
        pub_start_date="2024-01-01T00:00:00+00:00",
        pub_end_date="2024-07-19T00:00:00+00:00",
    )
    dataset = extractor.extract()

    assert len(dataset) == 1


def test_nvd_api_extractor_registered():
    """NVDAPIExtractor is registered as 'nvd_api'."""
    import vulnflow.extractors.nvd_api

    # Re-register by reloading the module (restore_registry fixture
    # clears the registry between tests, and cached module imports
    # don't re-execute the decorator).
    importlib.reload(vulnflow.extractors.nvd_api)

    from vulnflow.registry import Registry

    cls = Registry.get_extractor("nvd_api")
    assert cls.__name__ == "NVDAPIExtractor"


def test_nvd_api_extractor_requires_at_least_one_param():
    """NVDAPIExtractor raises if no query parameters provided."""
    from vulnflow.extractors.nvd_api import NVDAPIExtractor

    with pytest.raises(ValueError, match="At least one query parameter"):
        NVDAPIExtractor()


def test_nvd_api_extractor_provenance_label():
    """NVDAPIExtractor has descriptive provenance label."""
    from vulnflow.extractors.nvd_api import NVDAPIExtractor

    ext = NVDAPIExtractor(cve_id="CVE-2024-3094")
    assert "NVDAPIExtractor" in ext.provenance_label
    assert "nvd_api" in ext.provenance_label or "CVE-2024-3094" in ext.provenance_label


@patch("vulnflow.extractors.nvd_api.HttpFetcher")
def test_nvd_api_extractor_http_error(MockFetcher):
    """NVDAPIExtractor propagates HTTP errors from fetch_sync."""
    import httpx

    from vulnflow.extractors.nvd_api import NVDAPIExtractor

    instance = MockFetcher.return_value
    instance.fetch_sync.side_effect = httpx.ConnectError("Connection refused")

    extractor = NVDAPIExtractor(cve_id="CVE-2024-0001")
    with pytest.raises(httpx.ConnectError, match="Connection refused"):
        extractor.extract()


@patch("vulnflow.extractors.nvd_api.HttpFetcher")
def test_nvd_api_extractor_timings_populated(MockFetcher):
    """NVDAPIExtractor populates _timings dict after extract."""
    from vulnflow.extractors.nvd_api import NVDAPIExtractor

    response = _make_api_response(["CVE-2024-0001"])
    instance = MockFetcher.return_value
    instance.fetch_sync.return_value = (json.dumps(response).encode(), "json")

    extractor = NVDAPIExtractor(cve_id="CVE-2024-0001")
    extractor.extract()

    assert hasattr(extractor, "_timings")
    assert "fetch" in extractor._timings
    assert "populate" in extractor._timings
    assert extractor._timings["fetch"] >= 0
    assert extractor._timings["populate"] >= 0


@patch("vulnflow.extractors.nvd_api.HttpFetcher")
def test_nvd_api_extractor_provenance_date_range(MockFetcher):
    """NVDAPIExtractor sets date_min/date_max in provenance."""
    from vulnflow.extractors.nvd_api import NVDAPIExtractor

    response = _make_api_response(["CVE-2024-0001", "CVE-2024-0002"])
    instance = MockFetcher.return_value
    instance.fetch_sync.return_value = (json.dumps(response).encode(), "json")

    extractor = NVDAPIExtractor(keyword="test")
    dataset = extractor.extract()

    prov = dataset.provenance
    assert prov.get("date_min") is not None
    assert prov.get("date_max") is not None
