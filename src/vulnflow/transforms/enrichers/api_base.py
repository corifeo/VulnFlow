"""
API Enricher Base - Per-CVE API enrichment with rate limiting and retry

Provides:
- APIEnricher: Base class for per-CVE API enrichers

Layer: Transform/Enrichers
Dependencies: base, utils.http
Used by: Custom per-CVE enrichers
"""

from __future__ import annotations

import asyncio
from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from vulnflow.transforms.base import BaseEnricher
from vulnflow.utils.http import HttpError

if TYPE_CHECKING:
    from vulnflow.dataset import VulnDataset


class APIEnricher(BaseEnricher):
    """Base class for per-CVE API enrichers with rate limiting and retry.

    Unlike BulkEnrichers (KEV, EPSS) which download full datasets,
    APIEnrichers make individual API calls per CVE with concurrency
    control and exponential backoff retry.

    Subclasses MUST define:
    - name: str
    - OUTPUT_SCHEMA: list[FieldDefinition]
    - depends_on: list[str]
    - _fetch_for_cve(cve_id) -> dict | None
    - _apply_results(dataset, results) -> None

    Subclasses MUST call super().__init__() to register schema.
    """

    def __init__(
        self,
        max_parallel: int = 10,
        rate_limit: float = 1.0,
        max_retries: int = 3,
    ) -> None:
        """Initialize API enricher with concurrency and retry settings.

        Calls super().__init__() to chain through BaseTransform
        for schema validation and registration.

        Args:
            max_parallel: Maximum concurrent API requests
            rate_limit: Maximum requests per second
            max_retries: Maximum retry attempts per CVE on failure
        """
        super().__init__()
        self.max_parallel = max_parallel
        self.rate_limit = rate_limit
        self.max_retries = max_retries
        self._semaphore: asyncio.Semaphore | None = None

    @abstractmethod
    async def _fetch_for_cve(self, cve_id: str) -> dict[str, Any] | None:
        """Fetch data for a single CVE from an external API.

        Args:
            cve_id: CVE identifier (e.g. "CVE-2024-1234")

        Returns:
            Dict of enrichment data, or None if no data available
        """
        ...

    @abstractmethod
    def _apply_results(
        self,
        dataset: VulnDataset,
        results: list[dict[str, Any] | None],
    ) -> None:
        """Apply fetched results to dataset columns.

        Args:
            dataset: VulnDataset to update (modifies in place)
            results: List of results aligned with dataset rows (may contain None)
        """
        ...

    async def _fetch_with_retry(self, cve_id: str) -> dict[str, Any] | None:
        """Fetch data for a CVE with exponential backoff retry.

        Retries up to max_retries times on network errors
        (HttpError, OSError, TimeoutError) with exponential backoff.

        Args:
            cve_id: CVE identifier

        Returns:
            Dict of enrichment data, or None on final failure
        """
        for attempt in range(self.max_retries):
            try:
                return await self._fetch_for_cve(cve_id)
            except (HttpError, OSError, TimeoutError) as exc:
                if attempt < self.max_retries - 1:
                    delay = 2.0**attempt
                    self.logger.debug(
                        "Retry %d/%d for %s after error: %s",
                        attempt + 1,
                        self.max_retries,
                        cve_id,
                        exc,
                    )
                    await asyncio.sleep(delay)
                else:
                    self.logger.warning(
                        "Failed to fetch %s after %d retries: %s",
                        cve_id,
                        self.max_retries,
                        exc,
                    )
                    return None
        return None  # unreachable, but satisfies type checker

    async def _fetch_with_limits(self, cve_id: str) -> dict[str, Any] | None:
        """Fetch data for a CVE with rate limiting and concurrency control.

        Lazily creates semaphore on first call. Applies rate limiting
        delay before each request.

        Args:
            cve_id: CVE identifier

        Returns:
            Dict of enrichment data, or None on failure
        """
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_parallel)

        await asyncio.sleep(1.0 / self.rate_limit)
        async with self._semaphore:
            return await self._fetch_with_retry(cve_id)

    async def transform(self, dataset: VulnDataset) -> None:  # type: ignore[override]
        """Enrich dataset by fetching data for each CVE via API.

        Gets cve_id column, fetches data for all CVEs concurrently
        (with rate limiting), then applies results to dataset.

        Args:
            dataset: VulnDataset to enrich (modifies in place)
        """
        cve_ids: list[str] = list(dataset.get_column("cve_id"))

        results: list[dict[str, Any] | None] = list(
            await asyncio.gather(
                *(self._fetch_with_limits(cve_id) for cve_id in cve_ids)
            )
        )

        self._apply_results(dataset, results)
