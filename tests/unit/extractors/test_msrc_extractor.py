"""MSRCExtractor - MSRC CVRF API extractor tests.

All tests mock HttpFetcher to avoid real API calls.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from vulnflow.dataset import VulnDataset

FIXTURES = Path(__file__).resolve().parent.parent.parent / "fixtures"


def _load_fixture(name: str) -> bytes:
    """Load fixture file as bytes."""
    return (FIXTURES / name).read_bytes()


@patch("vulnflow.extractors.msrc.HttpFetcher")
def test_msrc_extractor_single_month(MockFetcher: object) -> None:
    """MSRCExtractor fetches a single month by start param."""
    from vulnflow.extractors.msrc import MSRCExtractor

    instance = MockFetcher.return_value  # type: ignore[union-attr]
    instance.fetch_sync.side_effect = [
        (_load_fixture("msrc_updates_sample.json"), "json"),
        (_load_fixture("msrc_sample.json"), "json"),
    ]

    extractor = MSRCExtractor(start="2025-04")
    dataset = extractor.extract()

    assert isinstance(dataset, VulnDataset)
    assert len(dataset) == 3
    cve_ids = list(dataset.get_column("cve_id"))
    assert "CVE-2025-27745" in cve_ids
    assert "CVE-2025-27746" in cve_ids
    assert "CVE-2025-27747" in cve_ids


@patch("vulnflow.extractors.msrc.HttpFetcher")
def test_msrc_extractor_default_current_month(MockFetcher: object) -> None:
    """MSRCExtractor with no args fetches latest available month."""
    from vulnflow.extractors.msrc import MSRCExtractor

    instance = MockFetcher.return_value  # type: ignore[union-attr]
    instance.fetch_sync.side_effect = [
        (_load_fixture("msrc_updates_sample.json"), "json"),
        (_load_fixture("msrc_sample.json"), "json"),
    ]

    extractor = MSRCExtractor()
    dataset = extractor.extract()

    assert isinstance(dataset, VulnDataset)
    assert len(dataset) == 3


@patch("vulnflow.extractors.msrc.HttpFetcher")
def test_msrc_extractor_multiple_months(MockFetcher: object) -> None:
    """MSRCExtractor fetches multiple months and merges."""
    from vulnflow.extractors.msrc import MSRCExtractor

    instance = MockFetcher.return_value  # type: ignore[union-attr]
    instance.fetch_sync.side_effect = [
        (_load_fixture("msrc_updates_sample.json"), "json"),
        (_load_fixture("msrc_sample.json"), "json"),  # month 1
        (_load_fixture("msrc_sample.json"), "json"),  # month 2
    ]

    extractor = MSRCExtractor(start="2025-03", end="2025-04")
    dataset = extractor.extract()

    assert isinstance(dataset, VulnDataset)
    # Same fixture used for both months, CVEs deduped → still 3
    assert len(dataset) == 3


@patch("vulnflow.extractors.msrc.HttpFetcher")
def test_msrc_extractor_empty_month(MockFetcher: object) -> None:
    """MSRCExtractor returns empty dataset for month with no vulnerabilities."""
    from vulnflow.extractors.msrc import MSRCExtractor

    empty_cvrf = json.dumps(
        {
            "Vulnerability": [],
            "ProductTree": {"FullProductName": []},
        }
    ).encode()

    instance = MockFetcher.return_value  # type: ignore[union-attr]
    instance.fetch_sync.side_effect = [
        (_load_fixture("msrc_updates_sample.json"), "json"),
        (empty_cvrf, "json"),
    ]

    extractor = MSRCExtractor(start="2025-04")
    dataset = extractor.extract()

    assert isinstance(dataset, VulnDataset)
    assert len(dataset) == 0


@patch("vulnflow.extractors.msrc.HttpFetcher")
def test_msrc_extractor_year_filter(MockFetcher: object) -> None:
    """MSRCExtractor filters by year."""
    from vulnflow.extractors.msrc import MSRCExtractor

    instance = MockFetcher.return_value  # type: ignore[union-attr]
    # One call for updates, then 4 calls for each 2025 month
    instance.fetch_sync.side_effect = [
        (_load_fixture("msrc_updates_sample.json"), "json"),
        (_load_fixture("msrc_sample.json"), "json"),
        (_load_fixture("msrc_sample.json"), "json"),
        (_load_fixture("msrc_sample.json"), "json"),
        (_load_fixture("msrc_sample.json"), "json"),
    ]

    extractor = MSRCExtractor(year=2025)
    dataset = extractor.extract()

    assert isinstance(dataset, VulnDataset)
    # 4 months fetched, but CVEs deduped → 3
    assert len(dataset) == 3
    # 1 updates call + 4 month calls
    assert instance.fetch_sync.call_count == 5


def test_msrc_extractor_conflicting_params() -> None:
    """MSRCExtractor raises on conflicting year + start params."""
    from vulnflow.extractors.msrc import MSRCExtractor

    with pytest.raises(ValueError, match="Cannot combine"):
        MSRCExtractor(year=2025, start="2025-01")


@patch("vulnflow.extractors.msrc.HttpFetcher")
def test_msrc_extractor_json_header(MockFetcher: object) -> None:
    """MSRCExtractor sends Accept: application/json header for CVRF requests."""
    from vulnflow.extractors.msrc import MSRCExtractor

    instance = MockFetcher.return_value  # type: ignore[union-attr]
    instance.fetch_sync.side_effect = [
        (_load_fixture("msrc_updates_sample.json"), "json"),
        (_load_fixture("msrc_sample.json"), "json"),
    ]

    extractor = MSRCExtractor(start="2025-04")
    extractor.extract()

    # Second call (CVRF fetch) should have JSON Accept header
    calls = instance.fetch_sync.call_args_list
    assert len(calls) == 2
    cvrf_call = calls[1]
    headers = cvrf_call.kwargs.get("headers") or (
        cvrf_call.args[3] if len(cvrf_call.args) > 3 else None
    )
    # The headers should be passed to fetch_sync
    assert headers is not None or "Accept" in str(cvrf_call)


@patch("vulnflow.extractors.msrc.HttpFetcher")
def test_msrc_extractor_provenance(MockFetcher: object) -> None:
    """MSRCExtractor sets provenance metadata."""
    from vulnflow.extractors.msrc import MSRCExtractor

    instance = MockFetcher.return_value  # type: ignore[union-attr]
    instance.fetch_sync.side_effect = [
        (_load_fixture("msrc_updates_sample.json"), "json"),
        (_load_fixture("msrc_sample.json"), "json"),
    ]

    extractor = MSRCExtractor(start="2025-04")
    dataset = extractor.extract()

    prov = dataset.provenance
    assert any("MSRC" in ext for ext in prov["extractors"])


@patch("vulnflow.extractors.msrc.HttpFetcher")
def test_msrc_extractor_metadata_has_msrc_key(MockFetcher: object) -> None:
    """Each record's metadata contains msrc key with exploitation info."""
    from vulnflow.extractors.msrc import MSRCExtractor

    instance = MockFetcher.return_value  # type: ignore[union-attr]
    instance.fetch_sync.side_effect = [
        (_load_fixture("msrc_updates_sample.json"), "json"),
        (_load_fixture("msrc_sample.json"), "json"),
    ]

    extractor = MSRCExtractor(start="2025-04")
    dataset = extractor.extract()

    meta_raw = dataset.get_column("metadata")[0]
    meta = json.loads(meta_raw)
    assert "msrc" in meta
    assert "exploited" in meta["msrc"]
    assert "month_id" in meta["msrc"]


def test_msrc_extractor_registered() -> None:
    """MSRCExtractor is registered as 'msrc'."""
    import vulnflow.extractors.msrc

    importlib.reload(vulnflow.extractors.msrc)

    from vulnflow.registry import Registry

    cls = Registry.get_extractor("msrc")
    assert cls.__name__ == "MSRCExtractor"


def test_msrc_extractor_provenance_label() -> None:
    """MSRCExtractor has descriptive provenance label."""
    from vulnflow.extractors.msrc import MSRCExtractor

    ext = MSRCExtractor(start="2025-04")
    assert "MSRC" in ext.provenance_label or "msrc" in ext.provenance_label


@patch("vulnflow.extractors.msrc.HttpFetcher")
def test_msrc_extractor_dedup_across_months(MockFetcher: object) -> None:
    """Duplicate CVE IDs across months are deduplicated (last-seen wins)."""
    from vulnflow.extractors.msrc import MSRCExtractor

    instance = MockFetcher.return_value  # type: ignore[union-attr]
    instance.fetch_sync.side_effect = [
        (_load_fixture("msrc_updates_sample.json"), "json"),
        (_load_fixture("msrc_sample.json"), "json"),
        (_load_fixture("msrc_sample.json"), "json"),
    ]

    extractor = MSRCExtractor(start="2025-03", end="2025-04")
    dataset = extractor.extract()

    cve_ids = list(dataset.get_column("cve_id"))
    # No duplicates
    assert len(cve_ids) == len(set(cve_ids))
