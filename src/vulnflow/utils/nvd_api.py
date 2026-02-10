"""NVD API Utilities - Shared helpers for NVD REST API 2.0 interactions.

Provides:
- resolve_api_key: Resolve API key from explicit arg or NVD_API_KEY env var
- get_sleep_interval: Rate-limit sleep duration based on API key presence
- chunk_date_range: Split date ranges into NVD-compliant 120-day windows
- build_query_params: Build NVD API query parameters from Python kwargs
- parse_api_page: Extract metadata and vulnerabilities from CVE API response
- parse_history_page: Extract metadata and changes from CVE History API response

Layer: Utility
Dependencies: None (pure functions, no vulnflow imports)
Used by: NVDAPIExtractor, NVDHistoryEnricher
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any

# ---------------------------------------------------------------------------
# NVD API endpoints
# ---------------------------------------------------------------------------
NVD_CVE_API_URL: str = "https://services.nvd.nist.gov/rest/json/cves/2.0"
NVD_HISTORY_API_URL: str = "https://services.nvd.nist.gov/rest/json/cvehistory/2.0"

# ---------------------------------------------------------------------------
# Pagination defaults
# ---------------------------------------------------------------------------
DEFAULT_RESULTS_PER_PAGE: int = 2000
HISTORY_RESULTS_PER_PAGE: int = 5000

# ---------------------------------------------------------------------------
# Date chunking
# ---------------------------------------------------------------------------
MAX_DATE_RANGE_DAYS: int = 120

# ---------------------------------------------------------------------------
# Rate limits: 50 req/30s with key, 5 req/30s without
# Simplified to fixed sleep intervals.
# Future: Replace with rolling-window rate limiter that tracks
# request timestamps in a deque and sleeps only when the window
# is full. This would allow bursting and better throughput.
# ---------------------------------------------------------------------------
_SLEEP_WITH_KEY: float = 0.6
_SLEEP_WITHOUT_KEY: float = 6.0

# ---------------------------------------------------------------------------
# NVD API date format
# ---------------------------------------------------------------------------
_NVD_DATE_FMT: str = "%Y-%m-%dT%H:%M:%S.%f"


# ===================================================================
# Public API
# ===================================================================


def resolve_api_key(api_key: str | None) -> str | None:
    """Resolve NVD API key: explicit value > NVD_API_KEY env var > None."""
    if api_key is not None:
        return api_key
    return os.environ.get("NVD_API_KEY")


def get_sleep_interval(api_key: str | None) -> float:
    """Return sleep interval in seconds based on API key presence."""
    if api_key is not None:
        return _SLEEP_WITH_KEY
    return _SLEEP_WITHOUT_KEY


def chunk_date_range(
    start: datetime,
    end: datetime,
    max_days: int = MAX_DATE_RANGE_DAYS,
) -> list[tuple[datetime, datetime]]:
    """Split a date range into chunks of at most *max_days* days.

    The NVD API rejects date ranges wider than 120 days.  This function
    produces contiguous, non-overlapping ``(chunk_start, chunk_end)`` pairs
    that cover the full ``[start, end]`` interval.

    Returns a single-element list when the range fits in one chunk.
    """
    chunks: list[tuple[datetime, datetime]] = []
    delta = timedelta(days=max_days)
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + delta, end)
        chunks.append((chunk_start, chunk_end))
        chunk_start = chunk_end
    return chunks


def build_query_params(
    *,
    cve_id: str | None = None,
    pub_start_date: datetime | str | None = None,
    pub_end_date: datetime | str | None = None,
    mod_start_date: datetime | str | None = None,
    mod_end_date: datetime | str | None = None,
    keyword: str | None = None,
    keyword_exact_match: bool = False,
    cpe_name: str | None = None,
    cvss_v3_severity: str | None = None,
    has_kev: bool = False,
    no_rejected: bool = False,
    results_per_page: int | None = None,
    start_index: int | None = None,
    extra_params: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build NVD API query parameters from Python-friendly kwargs.

    Maps snake_case keyword arguments to the camelCase parameter names
    expected by the NVD API.  Validates paired parameters (date ranges,
    keyword + exact match).

    Raises:
        ValueError: On invalid parameter combinations (unpaired dates,
            keywordExactMatch without keyword).
    """
    params: dict[str, str] = {}

    # -- scalar params --
    if cve_id is not None:
        params["cveId"] = cve_id

    # -- date pairs --
    _add_date_pair(params, "pubStartDate", "pubEndDate", pub_start_date, pub_end_date)
    _add_date_pair(params, "lastModStartDate", "lastModEndDate", mod_start_date, mod_end_date)

    # -- keyword search --
    if keyword_exact_match and keyword is None:
        msg = "keyword_exact_match requires keyword"
        raise ValueError(msg)
    if keyword is not None:
        params["keywordSearch"] = keyword
    if keyword_exact_match:
        params["keywordExactMatch"] = ""

    # -- CPE --
    if cpe_name is not None:
        params["cpeName"] = cpe_name

    # -- CVSS severity --
    if cvss_v3_severity is not None:
        params["cvssV3Severity"] = cvss_v3_severity

    # -- boolean flags (NVD expects the key present, value ignored) --
    if has_kev:
        params["hasKev"] = ""
    if no_rejected:
        params["noRejected"] = ""

    # -- pagination --
    if results_per_page is not None:
        params["resultsPerPage"] = str(results_per_page)
    if start_index is not None:
        params["startIndex"] = str(start_index)

    # -- escape hatch: merge arbitrary extra params last --
    if extra_params is not None:
        params.update(extra_params)

    return params


def parse_api_page(
    data: dict[str, Any],
) -> tuple[int, int, list[dict[str, Any]]]:
    """Extract pagination metadata and vulnerabilities from a CVE API page.

    Returns:
        (total_results, start_index, vulnerabilities) where
        *vulnerabilities* is the raw list of ``{cve: {...}}`` dicts.
    """
    total: int = data.get("totalResults", 0)
    start: int = data.get("startIndex", 0)
    vulns: list[dict[str, Any]] = data.get("vulnerabilities", [])
    return total, start, vulns


def parse_history_page(
    data: dict[str, Any],
) -> tuple[int, int, list[dict[str, Any]]]:
    """Extract pagination metadata and change records from a History API page.

    Returns:
        (total_results, start_index, changes) where *changes* is a flat
        list of change dicts (the inner ``change`` objects, unwrapped).
    """
    total: int = data.get("totalResults", 0)
    start: int = data.get("startIndex", 0)
    raw_changes: list[dict[str, Any]] = data.get("cveChanges", [])
    changes: list[dict[str, Any]] = [
        entry["change"] for entry in raw_changes if "change" in entry
    ]
    return total, start, changes


# ===================================================================
# Private helpers
# ===================================================================


def _format_date(value: datetime | str) -> str:
    """Format a datetime or string into the NVD API date format.

    If *value* is already a string it is returned as-is (the caller is
    responsible for correct formatting).
    """
    if isinstance(value, str):
        return value
    return value.strftime(_NVD_DATE_FMT)


def _add_date_pair(
    params: dict[str, str],
    start_key: str,
    end_key: str,
    start_val: datetime | str | None,
    end_val: datetime | str | None,
) -> None:
    """Validate and add a start/end date pair to *params*.

    Raises ``ValueError`` if only one of the pair is provided.
    """
    has_start = start_val is not None
    has_end = end_val is not None
    if has_start != has_end:
        msg = f"{start_key} and {end_key} must be provided together"
        raise ValueError(msg)
    if has_start and has_end:
        # Type narrowing: at this point both are not None
        assert start_val is not None
        assert end_val is not None
        params[start_key] = _format_date(start_val)
        params[end_key] = _format_date(end_val)
