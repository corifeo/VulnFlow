"""NVD API Extractor - Query-oriented NVD REST API 2.0 fetcher

Provides:
- NVDAPIExtractor: Fetches CVEs from the NVD API with pagination and date chunking

Layer: Extract
Dependencies: utils.http, utils.nvd, utils.nvd_api
Used by: dataset, pipeline, CacheBuilder
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import TYPE_CHECKING, Any

from vulnflow.extractors.base import BaseExtractor
from vulnflow.registry import Registry
from vulnflow.utils import json
from vulnflow.utils.http import HttpFetcher
from vulnflow.utils.nvd import parse_timestamp, populate_columns
from vulnflow.utils.nvd_api import (
    DEFAULT_RESULTS_PER_PAGE,
    NVD_CVE_API_URL,
    build_query_params,
    chunk_date_range,
    get_sleep_interval,
    parse_api_page,
    resolve_api_key,
)

if TYPE_CHECKING:
    from vulnflow.dataset import VulnDataset


@Registry.register_extractor("nvd_api")
class NVDAPIExtractor(BaseExtractor):
    """Fetch CVEs from the NVD REST API 2.0 with pagination and date chunking.

    Supports all NVD CVE API query parameters: single CVE lookup, keyword
    search, date ranges, CPE name, CVSS severity, KEV filter, and more.
    Date ranges exceeding 120 days are automatically split into chunks.

    Results are deduplicated by CVE ID (last-seen wins) to handle
    overlapping date chunks.

    Example:
        >>> ext = NVDAPIExtractor(cve_id="CVE-2024-3094")
        >>> ds = ext.extract()

        >>> ext = NVDAPIExtractor(keyword="log4j", no_rejected=True)
        >>> ds = ext.extract()

        >>> ext = NVDAPIExtractor(
        ...     pub_start_date="2024-01-01T00:00:00+00:00",
        ...     pub_end_date="2024-06-30T00:00:00+00:00",
        ... )
        >>> ds = ext.extract()
    """

    name: str = "nvd_api"
    format: str = "nvd_api"

    def __init__(
        self,
        *,
        cve_id: str | None = None,
        keyword: str | None = None,
        keyword_exact_match: bool = False,
        pub_start_date: str | datetime | None = None,
        pub_end_date: str | datetime | None = None,
        mod_start_date: str | datetime | None = None,
        mod_end_date: str | datetime | None = None,
        cpe_name: str | None = None,
        cvss_v3_severity: str | None = None,
        has_kev: bool = False,
        no_rejected: bool = False,
        results_per_page: int | None = None,
        params: dict[str, str] | None = None,
        api_key: str | None = None,
        cache_dir: str = ".cache/nvd_api",
        cache_ttl: int | None = None,
        timeout: float = 120.0,
    ) -> None:
        """Initialize NVD API extractor.

        At least one query parameter (cve_id, keyword, date range, cpe_name,
        cvss_v3_severity, has_kev, or params) must be provided.

        Args:
            cve_id: Specific CVE ID to fetch (e.g., "CVE-2024-3094")
            keyword: Keyword search string
            keyword_exact_match: Require exact keyword match (needs keyword)
            pub_start_date: Published date range start (ISO 8601 string or datetime)
            pub_end_date: Published date range end (ISO 8601 string or datetime)
            mod_start_date: Modified date range start (ISO 8601 string or datetime)
            mod_end_date: Modified date range end (ISO 8601 string or datetime)
            cpe_name: CPE 2.3 name to filter by
            cvss_v3_severity: CVSS v3 severity filter (LOW, MEDIUM, HIGH, CRITICAL)
            has_kev: Filter to CVEs in CISA KEV catalog
            no_rejected: Exclude rejected CVEs
            results_per_page: Results per API page (default 2000, max 2000)
            params: Escape hatch for arbitrary API parameters
            api_key: NVD API key (falls back to NVD_API_KEY env var)
            cache_dir: Directory for HTTP response caching
            cache_ttl: HTTP cache TTL in seconds (None disables caching)
            timeout: HTTP request timeout in seconds

        Raises:
            ValueError: If no query parameters are provided
        """
        super().__init__()

        # Validate at least one query param is provided
        has_query = any(
            [
                cve_id is not None,
                keyword is not None,
                pub_start_date is not None,
                mod_start_date is not None,
                cpe_name is not None,
                cvss_v3_severity is not None,
                has_kev,
                params,
            ]
        )
        if not has_query:
            msg = "At least one query parameter must be provided"
            raise ValueError(msg)

        # Store query parameters
        self._cve_id = cve_id
        self._keyword = keyword
        self._keyword_exact_match = keyword_exact_match
        self._pub_start_date = parse_timestamp(pub_start_date)
        self._pub_end_date = parse_timestamp(pub_end_date)
        self._mod_start_date = parse_timestamp(mod_start_date)
        self._mod_end_date = parse_timestamp(mod_end_date)
        self._cpe_name = cpe_name
        self._cvss_v3_severity = cvss_v3_severity
        self._has_kev = has_kev
        self._no_rejected = no_rejected
        self._results_per_page = results_per_page or DEFAULT_RESULTS_PER_PAGE
        self._extra_params = params

        # Connection settings
        self._api_key = resolve_api_key(api_key)
        self._cache_dir = cache_dir
        self._cache_ttl = cache_ttl
        self._timeout = timeout

    @property
    def label(self) -> str:
        """Short display name for progress reporting."""
        if self._cve_id:
            return self._cve_id
        if self._keyword:
            return f"nvd_api({self._keyword})"
        return "nvd_api"

    @property
    def provenance_label(self) -> str:
        """Human-readable label for provenance tracking."""
        if self._cve_id:
            return f"NVDAPIExtractor({self._cve_id})"
        if self._keyword:
            return f"NVDAPIExtractor(keyword={self._keyword})"
        return "NVDAPIExtractor(nvd_api)"

    def extract(self) -> VulnDataset:
        """Fetch CVEs from NVD API and return VulnDataset.

        Handles pagination within each date chunk. Date ranges exceeding
        120 days are automatically split. Results are deduplicated by CVE ID.

        Populates self._timings with per-phase durations (fetch, populate)
        for progress reporting.

        Returns:
            VulnDataset with columnar storage populated

        Raises:
            HttpError: On network or HTTP errors
        """
        from vulnflow.dataset import VulnDataset

        fetcher = HttpFetcher(
            cache_dir=self._cache_dir,
            timeout=self._timeout,
        )
        sleep_interval = get_sleep_interval(self._api_key)

        # Determine date chunks for pub_start/pub_end
        date_chunks = self._get_date_chunks()

        # Collect all vulnerability dicts across chunks and pages
        all_vulns: list[dict[str, Any]] = []

        with self._timed("fetch"):
            for chunk_start, chunk_end in date_chunks:
                chunk_vulns = self._fetch_chunk(
                    fetcher, chunk_start, chunk_end, sleep_interval
                )
                all_vulns.extend(chunk_vulns)

        # Deduplicate by CVE ID (last-seen wins)
        seen: dict[str, int] = {}
        unique_vulns: list[dict[str, Any]] = []
        for vuln in all_vulns:
            cve_id = vuln.get("cve", {}).get("id", "")
            if cve_id in seen:
                # Replace the earlier occurrence
                unique_vulns[seen[cve_id]] = vuln
            else:
                seen[cve_id] = len(unique_vulns)
                unique_vulns.append(vuln)

        if not unique_vulns:
            return VulnDataset()

        with self._timed("populate"):
            arrays = populate_columns(unique_vulns)

        if not arrays:
            return VulnDataset()

        dataset = VulnDataset()
        dataset._data = arrays
        dataset._record_provenance(extractor=self.provenance_label)
        self._record_date_provenance(dataset, arrays)

        return dataset

    def _get_date_chunks(self) -> list[tuple[datetime | None, datetime | None]]:
        """Determine date chunks for the query.

        If pub_start_date and pub_end_date span >120 days, splits into
        compliant chunks. Otherwise returns a single (None, None) chunk
        meaning "use the dates as-is in a single request".

        Returns:
            List of (start, end) tuples. (None, None) means no date
            chunking needed.
        """
        if self._pub_start_date is not None and self._pub_end_date is not None:
            chunks = chunk_date_range(self._pub_start_date, self._pub_end_date)
            if len(chunks) > 1:
                return chunks  # type: ignore[return-value]
        # No chunking needed - single pass with original params
        return [(None, None)]

    _MAX_PAGES: int = 50

    def _fetch_chunk(
        self,
        fetcher: HttpFetcher,
        chunk_start: datetime | None,
        chunk_end: datetime | None,
        sleep_interval: float,
    ) -> list[dict[str, Any]]:
        """Fetch all pages for a single date chunk.

        Args:
            fetcher: HTTP fetcher instance
            chunk_start: Override pub_start_date for this chunk (None = use original)
            chunk_end: Override pub_end_date for this chunk (None = use original)
            sleep_interval: Rate-limit sleep between requests

        Returns:
            List of vulnerability dicts from all pages in this chunk
        """
        vulns: list[dict[str, Any]] = []
        start_index = 0
        total_results: int | None = None
        pbar = None
        page_count = 0

        try:
            while True:
                page_count += 1
                if page_count > self._MAX_PAGES:
                    self.logger.warning(
                        "NVD API: hit max page limit (%d), stopping pagination",
                        self._MAX_PAGES,
                    )
                    break

                # Build query params for this page
                query_params = build_query_params(
                    cve_id=self._cve_id,
                    pub_start_date=chunk_start or self._pub_start_date,
                    pub_end_date=chunk_end or self._pub_end_date,
                    mod_start_date=self._mod_start_date,
                    mod_end_date=self._mod_end_date,
                    keyword=self._keyword,
                    keyword_exact_match=self._keyword_exact_match,
                    cpe_name=self._cpe_name,
                    cvss_v3_severity=self._cvss_v3_severity,
                    has_kev=self._has_kev,
                    no_rejected=self._no_rejected,
                    results_per_page=self._results_per_page,
                    start_index=start_index,
                    extra_params=self._extra_params,
                )

                # Build URL with query string
                url = self._build_url(query_params)

                self.logger.info(
                    "NVD API: fetching page %d (startIndex=%d)",
                    page_count,
                    start_index,
                )

                # Fetch the page
                content, _ = fetcher.fetch_sync(
                    url,
                    cache_ttl=self._cache_ttl,
                )
                data: dict[str, Any] = json.loads(content)
                page_total, page_start, page_vulns = parse_api_page(data)

                vulns.extend(page_vulns)

                # Update total on every page (API may report different totals)
                if total_results is None:
                    total_results = page_total
                    # Create progress bar for multi-page fetches
                    if total_results > len(page_vulns):
                        pbar = self._progress(
                            total=total_results,
                            unit="CVEs",
                        )
                        pbar.update(len(page_vulns))
                else:
                    total_results = page_total
                    if pbar is not None:
                        pbar.total = total_results
                        pbar.update(len(page_vulns))

                # Log per-page progress
                fetched_so_far = start_index + len(page_vulns)
                self.logger.info(
                    "NVD API: fetched %d/%d CVEs (page %d)",
                    fetched_so_far,
                    total_results,
                    page_count,
                )

                # Check if we have all results
                if fetched_so_far >= total_results or len(page_vulns) == 0:
                    break

                # Advance to next page
                start_index = fetched_so_far

                # Rate limit between pages
                time.sleep(sleep_interval)
        finally:
            if pbar is not None:
                pbar.close()

        return vulns

    @staticmethod
    def _build_url(params: dict[str, str]) -> str:
        """Build full NVD API URL with query parameters.

        Args:
            params: Query parameter dict

        Returns:
            Full URL string with encoded query parameters
        """
        from urllib.parse import urlencode

        if not params:
            return NVD_CVE_API_URL
        return f"{NVD_CVE_API_URL}?{urlencode(params)}"
