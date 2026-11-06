"""MSRC Extractor - Fetch Patch Tuesday vulnerabilities from MSRC CVRF API v3.0.

Provides:
- MSRCExtractor: Fetches CVEs from monthly MSRC security updates

Layer: Extract
Dependencies: utils.http, utils.msrc, utils.json
Used by: dataset, pipeline, CacheBuilder

Note: MSRC and NVD datasets should be used independently or combined
with awareness that the current merge is lossy (full-record replacement,
last-seen wins). Field-level merge for multi-source CVE dedup is planned.
# TODO: field-level merge for multi-source CVE dedup
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vulnflow.exceptions import ExtractorError
from vulnflow.extractors.base import BaseExtractor
from vulnflow.registry import Registry
from vulnflow.utils import json
from vulnflow.utils.http import HttpError, HttpFetcher
from vulnflow.utils.msrc import resolve_month_ids

if TYPE_CHECKING:
    from vulnflow.dataset import VulnDataset

MSRC_API_BASE = "https://api.msrc.microsoft.com/cvrf/v3.0"
MSRC_UPDATES_URL = f"{MSRC_API_BASE}/updates"
_JSON_HEADERS = {"Accept": "application/json"}


@Registry.register_extractor("msrc")
class MSRCExtractor(BaseExtractor):
    """Fetch Patch Tuesday CVEs from the MSRC CVRF API v3.0.

    Supports filtering by year, month range, or current month (default).
    Results are deduplicated by CVE ID (last-seen wins) when fetching
    multiple months.

    Example:
        >>> ext = MSRCExtractor(start="2025-04")
        >>> ds = ext.extract()

        >>> ext = MSRCExtractor(year=2025)
        >>> ds = ext.extract()

        >>> ext = MSRCExtractor()  # current month
        >>> ds = ext.extract()
    """

    name: str = "msrc"
    format: str = "msrc_cvrf"

    def __init__(
        self,
        *,
        year: int | list[int] | None = None,
        start: str | None = None,
        end: str | None = None,
        cache_dir: str = ".cache/msrc",
        cache_ttl: int | None = None,
        timeout: float = 120.0,
    ) -> None:
        """Initialize MSRC extractor.

        Args:
            year: Filter to specific year(s). All available months in those years.
            start: Start month in YYYY-MM format (inclusive). Single month if no end.
            end: End month in YYYY-MM format (inclusive). Requires start.
            cache_dir: Directory for HTTP response caching
            cache_ttl: HTTP cache TTL in seconds (None disables caching)
            timeout: HTTP request timeout in seconds

        Raises:
            ValueError: If year and start/end are combined, or end without start
        """
        super().__init__()

        # Validate conflicting params early (before any API calls)
        if year is not None and start is not None:
            msg = "Cannot combine 'year' and 'start'/'end' parameters"
            raise ValueError(msg)
        if end is not None and start is None:
            msg = "'end' requires 'start' parameter"
            raise ValueError(msg)

        self._year = year
        self._start = start
        self._end = end
        self._cache_dir = cache_dir
        self._cache_ttl = cache_ttl
        self._timeout = timeout

    @property
    def label(self) -> str:
        """Short display name for progress reporting."""
        cls = type(self).__name__
        if self._start:
            if self._end:
                return f"{cls}({self._start}..{self._end})"
            return f"{cls}({self._start})"
        if self._year:
            return f"{cls}({self._year})"
        return f"{cls}()"

    @property
    def provenance_label(self) -> str:
        """Human-readable label for provenance tracking."""
        if self._start:
            if self._end:
                return f"MSRCExtractor({self._start}..{self._end})"
            return f"MSRCExtractor({self._start})"
        if self._year:
            return f"MSRCExtractor(year={self._year})"
        return "MSRCExtractor(current)"

    def extract(self) -> VulnDataset:
        """Fetch CVEs from MSRC API and return VulnDataset.

        1. Fetches /updates to get available month IDs
        2. Filters by year/start/end params
        3. Fetches CVRF document for each month
        4. Converts to columnar arrays
        5. Deduplicates by CVE ID (last-seen wins)

        Returns:
            VulnDataset with columnar storage populated

        Raises:
            ExtractorError: On HTTP failures or malformed CVRF JSON
        """
        from vulnflow.dataset import VulnDataset

        fetcher = HttpFetcher(
            cache_dir=self._cache_dir,
            timeout=self._timeout,
        )

        # Step 1–3: Fetch available months, resolve, and fetch CVRF
        all_vulns: list[tuple[dict[str, Any], str]] = []  # (vuln_dict, month_id)
        with self._timed("fetch"):
            try:
                available_ids = self._fetch_available_months(fetcher)
            except HttpError as exc:
                raise ExtractorError(
                    f"Failed to fetch MSRC updates list: {exc}"
                ) from exc

            try:
                month_ids = resolve_month_ids(
                    year=self._year,
                    start=self._start,
                    end=self._end,
                    available=available_ids,
                )
            except ValueError as exc:
                raise ExtractorError(str(exc)) from exc

            self.logger.info(
                "MSRC: fetching %d month(s): %s", len(month_ids), month_ids
            )

            with self._progress(total=len(month_ids), unit="months") as pbar:
                for mid in month_ids:
                    try:
                        vulns = self._fetch_month(fetcher, mid)
                        for v in vulns:
                            all_vulns.append((v, mid))
                    except HttpError as exc:
                        if len(month_ids) > 1:
                            self.logger.warning(
                                "MSRC: skipping month %s due to HTTP error: %s",
                                mid,
                                exc,
                            )
                            pbar.update(1)
                            continue
                        raise ExtractorError(
                            f"Failed to fetch MSRC data for {mid}: {exc}"
                        ) from exc
                    except (KeyError, json.JSONDecodeError) as exc:
                        self.logger.warning(
                            "MSRC: skipping month %s due to parse error: %s", mid, exc
                        )
                        pbar.update(1)
                        continue
                    pbar.update(1)

        if not all_vulns:
            return VulnDataset()

        # Step 4: Deduplicate by CVE ID (last-seen wins)
        seen: dict[str, int] = {}
        unique_vulns: list[tuple[dict[str, Any], str]] = []
        for vuln, mid in all_vulns:
            cve_id = vuln.get("CVE", "")
            if cve_id in seen:
                unique_vulns[seen[cve_id]] = (vuln, mid)
            else:
                seen[cve_id] = len(unique_vulns)
                unique_vulns.append((vuln, mid))

        # Step 5: Convert to columnar arrays
        # Group by month_id for populate_msrc_columns
        with self._timed("populate"):
            from vulnflow.utils.msrc import populate_msrc_columns as _populate

            # Process all vulns, passing month_id per vuln
            # We need to handle mixed month_ids, so process per-month and merge
            month_groups: dict[str, list[dict[str, Any]]] = {}
            for vuln, mid in unique_vulns:
                month_groups.setdefault(mid, []).append(vuln)

            datasets: list[VulnDataset] = []
            for mid, vulns in month_groups.items():
                arrays = _populate(vulns, mid)
                if arrays:
                    ds = VulnDataset()
                    ds._data = arrays
                    datasets.append(ds)

        if not datasets:
            return VulnDataset()

        result = VulnDataset.merge(datasets) if len(datasets) > 1 else datasets[0]
        result._record_provenance(extractor=self.provenance_label)
        return result

    def _fetch_available_months(self, fetcher: HttpFetcher) -> list[str]:
        """Fetch /updates and return list of available month IDs.

        Args:
            fetcher: HTTP fetcher instance

        Returns:
            List of month IDs (e.g., ["2025-Jan", "2025-Feb", ...])
        """
        content, _ = fetcher.fetch_sync(
            MSRC_UPDATES_URL,
            cache_ttl=self._cache_ttl,
        )
        data: dict[str, Any] = json.loads(content)
        return [entry["ID"] for entry in data.get("value", [])]

    def _fetch_month(self, fetcher: HttpFetcher, month_id: str) -> list[dict[str, Any]]:
        """Fetch CVRF document for a single month.

        Args:
            fetcher: HTTP fetcher instance
            month_id: MSRC month ID (e.g., "2025-Apr")

        Returns:
            List of vulnerability dicts from the CVRF document
        """
        url = f"{MSRC_API_BASE}/cvrf/{month_id}"
        content, _ = fetcher.fetch_sync(
            url,
            cache_ttl=self._cache_ttl,
            headers=_JSON_HEADERS,
        )
        data: dict[str, Any] = json.loads(content)
        vulns: list[dict[str, Any]] = data.get("Vulnerability", [])
        self.logger.info("MSRC: fetched %d CVEs from %s", len(vulns), month_id)
        return vulns
