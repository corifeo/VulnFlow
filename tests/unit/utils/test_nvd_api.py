"""NVD API utilities - date chunking, API key resolution, param building."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest


def test_resolve_api_key_explicit():
    """Explicit api_key takes precedence over env var."""
    from vulnflow.utils.nvd_api import resolve_api_key
    assert resolve_api_key("my-key") == "my-key"


def test_resolve_api_key_env_var(monkeypatch):
    """Falls back to NVD_API_KEY env var."""
    from vulnflow.utils.nvd_api import resolve_api_key
    monkeypatch.setenv("NVD_API_KEY", "env-key")
    assert resolve_api_key(None) == "env-key"


def test_resolve_api_key_none(monkeypatch):
    """Returns None when no key available."""
    from vulnflow.utils.nvd_api import resolve_api_key
    monkeypatch.delenv("NVD_API_KEY", raising=False)
    assert resolve_api_key(None) is None


def test_get_sleep_interval_with_key():
    """With API key: ~0.6s between requests."""
    from vulnflow.utils.nvd_api import get_sleep_interval
    assert get_sleep_interval("key") == pytest.approx(0.6, abs=0.1)


def test_get_sleep_interval_without_key():
    """Without API key: ~6.0s between requests."""
    from vulnflow.utils.nvd_api import get_sleep_interval
    assert get_sleep_interval(None) == pytest.approx(6.0, abs=0.1)


def test_chunk_date_range_within_120_days():
    """Date range within 120 days returns single chunk."""
    from vulnflow.utils.nvd_api import chunk_date_range
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 3, 1, tzinfo=UTC)
    chunks = chunk_date_range(start, end)
    assert len(chunks) == 1
    assert chunks[0] == (start, end)


def test_chunk_date_range_exceeds_120_days():
    """Date range over 120 days is split into chunks."""
    from vulnflow.utils.nvd_api import chunk_date_range
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 12, 31, tzinfo=UTC)
    chunks = chunk_date_range(start, end)
    assert len(chunks) > 1
    assert chunks[0][0] == start
    assert chunks[-1][1] == end
    for i in range(len(chunks) - 1):
        assert chunks[i][1] == chunks[i + 1][0]


def test_chunk_date_range_exact_120_days():
    """Exactly 120 days returns single chunk."""
    from vulnflow.utils.nvd_api import chunk_date_range
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = start + timedelta(days=120)
    chunks = chunk_date_range(start, end)
    assert len(chunks) == 1


def test_build_query_params_cve_id():
    """build_query_params maps cve_id to cveId."""
    from vulnflow.utils.nvd_api import build_query_params
    params = build_query_params(cve_id="CVE-2024-3094")
    assert params["cveId"] == "CVE-2024-3094"


def test_build_query_params_date_range():
    """build_query_params formats dates as ISO-8601."""
    from vulnflow.utils.nvd_api import build_query_params
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 6, 30, tzinfo=UTC)
    params = build_query_params(pub_start_date=start, pub_end_date=end)
    assert "pubStartDate" in params
    assert "pubEndDate" in params


def test_build_query_params_string_dates():
    """build_query_params accepts string dates too."""
    from vulnflow.utils.nvd_api import build_query_params
    params = build_query_params(
        pub_start_date="2024-01-01T00:00:00.000",
        pub_end_date="2024-06-30T00:00:00.000",
    )
    assert "pubStartDate" in params
    assert "pubEndDate" in params


def test_build_query_params_boolean_flags():
    """build_query_params includes boolean flags without values."""
    from vulnflow.utils.nvd_api import build_query_params
    params = build_query_params(has_kev=True, no_rejected=True)
    assert "hasKev" in params
    assert "noRejected" in params


def test_build_query_params_keyword_exact_match():
    """keywordExactMatch requires keyword."""
    from vulnflow.utils.nvd_api import build_query_params
    params = build_query_params(keyword="log4j", keyword_exact_match=True)
    assert params["keywordSearch"] == "log4j"
    assert "keywordExactMatch" in params


def test_build_query_params_rejects_keyword_exact_without_keyword():
    """keywordExactMatch without keyword raises ValueError."""
    from vulnflow.utils.nvd_api import build_query_params
    with pytest.raises(ValueError, match="keyword_exact_match requires keyword"):
        build_query_params(keyword_exact_match=True)


def test_build_query_params_merge_escape_hatch():
    """Extra params dict is merged, overriding explicit kwargs."""
    from vulnflow.utils.nvd_api import build_query_params
    params = build_query_params(
        cve_id="CVE-2024-0001",
        extra_params={"cveTag": "disputed"},
    )
    assert params["cveId"] == "CVE-2024-0001"
    assert params["cveTag"] == "disputed"


def test_build_query_params_unpaired_pub_dates():
    """pub_start_date without pub_end_date raises ValueError."""
    from vulnflow.utils.nvd_api import build_query_params
    start = datetime(2024, 1, 1, tzinfo=UTC)
    with pytest.raises(ValueError, match="must be provided together"):
        build_query_params(pub_start_date=start)


def test_parse_api_page_extracts_metadata():
    """parse_api_page extracts totalResults, startIndex, vulnerabilities."""
    from vulnflow.utils.nvd_api import parse_api_page
    data = {
        "totalResults": 100,
        "startIndex": 0,
        "resultsPerPage": 2,
        "vulnerabilities": [
            {"cve": {"id": "CVE-2024-0001"}},
            {"cve": {"id": "CVE-2024-0002"}},
        ],
    }
    total, start, vulns = parse_api_page(data)
    assert total == 100
    assert start == 0
    assert len(vulns) == 2


def test_parse_history_page_extracts_changes():
    """parse_history_page extracts totalResults, startIndex, cveChanges."""
    from vulnflow.utils.nvd_api import parse_history_page
    data = {
        "totalResults": 50,
        "startIndex": 0,
        "resultsPerPage": 2,
        "cveChanges": [
            {
                "change": {
                    "cveId": "CVE-2024-0001",
                    "eventName": "CVE Modified",
                    "cveChangeId": "abc-123",
                    "created": "2024-06-01T10:00:00.000",
                    "details": [{"action": "Changed", "type": "Description"}],
                }
            },
        ],
    }
    total, start, changes = parse_history_page(data)
    assert total == 50
    assert start == 0
    assert len(changes) == 1
    assert changes[0]["cveId"] == "CVE-2024-0001"
