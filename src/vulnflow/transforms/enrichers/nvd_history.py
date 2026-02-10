"""NVD History Enricher - CVE change history from NVD REST API 2.0

Provides:
- NVDHistoryEnricher: Adds NVD CVE change history to existing datasets

Layer: Transform/Enrichers
Dependencies: dataset, utils.http, utils.nvd_api
Used by: dataset.transform(), pipeline

Two explicit modes:
- per_cve: Fetch history for each CVE ID individually (small datasets)
- date_range: Bulk fetch by date range, filter to dataset CVEs (large datasets)

Future: auto-select mode that picks per_cve vs date_range based on
dataset size and date spread heuristics.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlencode

import numpy as np

if TYPE_CHECKING:
    from vulnflow.dataset import VulnDataset

from vulnflow.registry import FieldDefinition, register_transform
from vulnflow.transforms.base import BaseEnricher
from vulnflow.utils import json
from vulnflow.utils.http import HttpError, HttpFetcher
from vulnflow.utils.nvd_api import (
    HISTORY_RESULTS_PER_PAGE,
    NVD_HISTORY_API_URL,
    chunk_date_range,
    get_sleep_interval,
    parse_history_page,
    resolve_api_key,
)

logger = logging.getLogger(__name__)


def _to_datetime(value: str | datetime | None) -> datetime | None:
    """Convert a string or datetime to a timezone-aware datetime.

    Accepts ISO 8601 strings (with or without timezone) and datetime objects.
    Returns None for None input.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


@register_transform(default=False)
class NVDHistoryEnricher(BaseEnricher):
    """Enrich dataset with NVD CVE change history.

    Fetches change history from the NVD CVE History API 2.0 and stores
    it as a JSON-encoded array in the ``nvd_changes`` column.

    Two modes:
    - ``per_cve``: Fetch history for each CVE ID individually.
      Best for small datasets or targeted lookups.
    - ``date_range``: Bulk fetch changes in a date window, then filter
      to CVEs present in the dataset. Best for large datasets.
    """

    name: str = "NVDHistoryEnricher"
    depends_on: list[str] = []

    OUTPUT_SCHEMA: list[FieldDefinition] = [
        FieldDefinition(
            "nvd_changes",
            "str",
            "NVD change history array (JSON)",
            "NVDHistoryEnricher",
            is_json=True,
        ),
    ]

    def __init__(
        self,
        *,
        mode: Literal["per_cve", "date_range"] = "per_cve",
        change_start_date: str | datetime | None = None,
        change_end_date: str | datetime | None = None,
        api_key: str | None = None,
        cache_dir: str = ".cache/nvd_history",
        cache_ttl: int | None = None,
        timeout: float = 120.0,
        results_per_page: int | None = None,
    ) -> None:
        """Initialize NVD History enricher.

        Args:
            mode: Fetch strategy — "per_cve" or "date_range".
            change_start_date: Start of date range (required for date_range mode).
            change_end_date: End of date range (required for date_range mode).
            api_key: NVD API key (falls back to NVD_API_KEY env var).
            cache_dir: Directory for HTTP response caching.
            cache_ttl: HTTP cache TTL in seconds (None disables caching).
            timeout: HTTP request timeout in seconds.
            results_per_page: Results per API page (default 5000 for history).

        Raises:
            ValueError: If mode is date_range but dates are missing.
        """
        super().__init__()

        self._mode = mode
        self._change_start = _to_datetime(change_start_date)
        self._change_end = _to_datetime(change_end_date)

        if mode == "date_range" and (self._change_start is None or self._change_end is None):
            msg = "change_start_date and change_end_date required for date_range mode"
            raise ValueError(msg)

        self._api_key = resolve_api_key(api_key)
        self._cache_dir = cache_dir
        self._cache_ttl = cache_ttl
        self._timeout = timeout
        self._results_per_page = results_per_page or HISTORY_RESULTS_PER_PAGE

    async def transform(self, dataset: VulnDataset) -> None:  # type: ignore[override]
        """Enrich dataset with NVD change history.

        Dispatches to per_cve or date_range strategy based on mode.

        Args:
            dataset: VulnDataset to enrich (modifies in place)
        """
        if self._mode == "per_cve":
            await self._transform_per_cve(dataset)
        else:
            await self._transform_date_range(dataset)

    # ------------------------------------------------------------------
    # Per-CVE mode
    # ------------------------------------------------------------------

    async def _transform_per_cve(self, dataset: VulnDataset) -> None:
        """Fetch history for each CVE ID individually."""
        fetcher = HttpFetcher(
            cache_dir=self._cache_dir,
            timeout=self._timeout,
        )
        sleep_interval = get_sleep_interval(self._api_key)

        cve_ids = dataset.get_column("cve_id")
        n = len(dataset)
        nvd_changes = np.array(["[]"] * n, dtype=object)

        for i, cve_id in enumerate(cve_ids):
            try:
                changes = await self._fetch_history_for_cve(fetcher, str(cve_id))
                nvd_changes[i] = json.dumps(changes)
            except (HttpError, Exception) as exc:
                logger.warning(
                    "Failed to fetch history for %s: %s", cve_id, exc
                )
                nvd_changes[i] = "[]"

            # Rate limit between requests (skip after last CVE)
            if i < n - 1:
                await asyncio.sleep(sleep_interval)

        dataset.set_column("nvd_changes", nvd_changes)

    async def _fetch_history_for_cve(
        self,
        fetcher: HttpFetcher,
        cve_id: str,
    ) -> list[dict[str, Any]]:
        """Fetch all change history pages for a single CVE.

        Args:
            fetcher: HTTP fetcher instance
            cve_id: CVE ID to query

        Returns:
            List of change dicts for this CVE
        """
        all_changes: list[dict[str, Any]] = []
        start_index = 0
        total_results: int | None = None

        while True:
            params: dict[str, str] = {
                "cveId": cve_id,
                "resultsPerPage": str(self._results_per_page),
                "startIndex": str(start_index),
            }
            url = self._build_url(params)

            content, _ = await fetcher.fetch(url, self._cache_ttl)
            data: dict[str, Any] = json.loads(content)
            page_total, _page_start, page_changes = parse_history_page(data)

            all_changes.extend(page_changes)

            if total_results is None:
                total_results = page_total

            fetched_so_far = start_index + len(page_changes)
            if fetched_so_far >= total_results or len(page_changes) == 0:
                break

            start_index = fetched_so_far
            await asyncio.sleep(get_sleep_interval(self._api_key))

        return all_changes

    # ------------------------------------------------------------------
    # Date-range mode
    # ------------------------------------------------------------------

    async def _transform_date_range(self, dataset: VulnDataset) -> None:
        """Bulk fetch changes by date range, filter to dataset CVEs."""
        assert self._change_start is not None
        assert self._change_end is not None

        fetcher = HttpFetcher(
            cache_dir=self._cache_dir,
            timeout=self._timeout,
        )
        sleep_interval = get_sleep_interval(self._api_key)

        # Collect all changes across date chunks
        all_changes: list[dict[str, Any]] = []
        date_chunks = chunk_date_range(self._change_start, self._change_end)

        for chunk_start, chunk_end in date_chunks:
            chunk_changes = await self._fetch_history_chunk(
                fetcher, chunk_start, chunk_end, sleep_interval
            )
            all_changes.extend(chunk_changes)

        # Group changes by CVE ID
        changes_by_cve: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for change in all_changes:
            cve_id = change.get("cveId", "")
            changes_by_cve[cve_id].append(change)

        # Map to dataset rows
        cve_ids = dataset.get_column("cve_id")
        n = len(dataset)
        nvd_changes = np.array(["[]"] * n, dtype=object)

        for i, cve_id in enumerate(cve_ids):
            cve_str = str(cve_id)
            if cve_str in changes_by_cve:
                nvd_changes[i] = json.dumps(changes_by_cve[cve_str])

        dataset.set_column("nvd_changes", nvd_changes)

    async def _fetch_history_chunk(
        self,
        fetcher: HttpFetcher,
        chunk_start: datetime,
        chunk_end: datetime,
        sleep_interval: float,
    ) -> list[dict[str, Any]]:
        """Fetch all pages of history changes for a date chunk.

        Args:
            fetcher: HTTP fetcher instance
            chunk_start: Start of date window
            chunk_end: End of date window
            sleep_interval: Rate-limit sleep between requests

        Returns:
            List of change dicts from all pages in this chunk
        """
        fmt = "%Y-%m-%dT%H:%M:%S.%f"
        all_changes: list[dict[str, Any]] = []
        start_index = 0
        total_results: int | None = None

        while True:
            params: dict[str, str] = {
                "changeStartDate": chunk_start.strftime(fmt),
                "changeEndDate": chunk_end.strftime(fmt),
                "resultsPerPage": str(self._results_per_page),
                "startIndex": str(start_index),
            }
            url = self._build_url(params)

            content, _ = await fetcher.fetch(url, self._cache_ttl)
            data: dict[str, Any] = json.loads(content)
            page_total, _page_start, page_changes = parse_history_page(data)

            all_changes.extend(page_changes)

            if total_results is None:
                total_results = page_total

            fetched_so_far = start_index + len(page_changes)
            if fetched_so_far >= total_results or len(page_changes) == 0:
                break

            start_index = fetched_so_far
            await asyncio.sleep(sleep_interval)

        return all_changes

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_url(params: dict[str, str]) -> str:
        """Build full NVD History API URL with query parameters.

        Args:
            params: Query parameter dict

        Returns:
            Full URL string with encoded query parameters
        """
        if not params:
            return NVD_HISTORY_API_URL
        return f"{NVD_HISTORY_API_URL}?{urlencode(params)}"
