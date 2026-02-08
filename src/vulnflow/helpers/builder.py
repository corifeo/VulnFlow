"""
CacheBuilder - Commodity class for building NVD vulnerability libraries

Wraps VulnPipeline with NVD-specific URL knowledge for one-line library
construction. Supports multiple modes: all years, specific years, modified
feed only, recent feed only.

Layer: User-facing helpers
Dependencies: pipeline, extractors.nvd
Used by: Notebooks, user scripts
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vulnflow.dataset import VulnDataset
    from vulnflow.extractors.base import BaseExtractor
    from vulnflow.transforms.base import BaseTransform

logger = logging.getLogger(__name__)

NVD_BASE_URL = "https://nvd.nist.gov/feeds/json/cve/2.0"


class CacheBuilder:
    """Build NVD vulnerability libraries with parallel extraction.

    Commodity class that wraps VulnPipeline for one-line library building.
    Knows NVD feed URL structure internally - no separate download step needed.

    Modes:
        CacheBuilder()                          # all years 2002-current
        CacheBuilder(years=[2024, 2025])        # specific years
        CacheBuilder(exclude_years=[2002])      # all minus excluded
        CacheBuilder(include_modified=True)      # any mode + Modified feed
        CacheBuilder(modified_only=True)         # Modified feed only
        CacheBuilder(recent_only=True)           # Recent feed only

    Example:
        >>> from vulnflow.helpers import CacheBuilder
        >>> ds = CacheBuilder(years=[2024, 2025]).build()
        >>> ds.to_parquet("data/library.parquet")
    """

    NVD_BASE_URL: str = NVD_BASE_URL

    def __init__(
        self,
        years: list[int] | None = None,
        exclude_years: list[int] | None = None,
        include_modified: bool = False,
        modified_only: bool = False,
        recent_only: bool = False,
        quiet: bool = False,
    ) -> None:
        """Initialize CacheBuilder with feed selection parameters.

        Args:
            years: Specific years to include (default: all 2002-current).
            exclude_years: Years to exclude from full range.
            include_modified: Add Modified feed to year-based modes.
            modified_only: Build from Modified feed only (quick update).
            recent_only: Build from Recent feed only.
            quiet: Suppress all progress output during build.

        Raises:
            ValueError: If mutually exclusive parameters are combined.
        """
        self._validate_params(
            years, exclude_years, include_modified, modified_only, recent_only
        )
        self._urls = self._resolve_urls(
            years, exclude_years, include_modified, modified_only, recent_only
        )
        self._quiet = quiet

    @staticmethod
    def _validate_params(
        years: list[int] | None,
        exclude_years: list[int] | None,
        include_modified: bool,
        modified_only: bool,
        recent_only: bool,
    ) -> None:
        """Validate parameter combinations."""
        if years is not None and exclude_years is not None:
            raise ValueError("years and exclude_years are mutually exclusive")
        if modified_only and (years is not None or exclude_years is not None):
            raise ValueError(
                "modified_only and years/exclude_years are mutually exclusive"
            )
        if recent_only and (years is not None or exclude_years is not None):
            raise ValueError(
                "recent_only and years/exclude_years are mutually exclusive"
            )
        if modified_only and recent_only:
            raise ValueError("modified_only and recent_only are mutually exclusive")

    @classmethod
    def _resolve_urls(
        cls,
        years: list[int] | None,
        exclude_years: list[int] | None,
        include_modified: bool,
        modified_only: bool,
        recent_only: bool,
    ) -> list[str]:
        """Resolve NVD feed URLs from parameters."""
        if modified_only:
            return [cls.feed_url("modified")]
        if recent_only:
            return [cls.feed_url("recent")]

        # Determine year list
        current_year = datetime.now().year
        if years is not None:
            year_list = sorted(years)
        elif exclude_years is not None:
            all_years = set(range(2002, current_year + 1))
            year_list = sorted(all_years - set(exclude_years))
        else:
            year_list = list(range(2002, current_year + 1))

        urls = [cls.feed_url(y) for y in year_list]

        if include_modified:
            urls.append(cls.feed_url("modified"))

        return urls

    def build(
        self,
        timeout: float = 120.0,
        cvss: bool = True,
        epss: bool = True,
        kev: bool = True,
    ) -> VulnDataset:
        """Build vulnerability dataset with concurrent extraction.

        Uses ThreadPoolExecutor for concurrent feed extraction (no subprocess
        overhead). Enriches with EPSS/KEV by default (bulk CSV downloads,
        cached locally for 24h).

        Args:
            timeout: HTTP timeout in seconds per feed.
            cvss: Flatten CVSS scores to scalar columns (default: True).
            epss: Enrich with EPSS exploitation probabilities (default: True).
            kev: Enrich with CISA KEV catalog data (default: True).

        Returns:
            Merged VulnDataset with all extracted records.
        """
        from vulnflow.extractors.nvd import NVDExtractor
        from vulnflow.pipeline import VulnPipeline
        from vulnflow.utils.progress import _suppress_progress

        logger.info(
            "Building library from %d feeds",
            len(self._urls),
        )

        # Suppress progress output when quiet mode is enabled
        old_tqdm_disable = os.environ.get("TQDM_DISABLE")
        suppress_token = None
        if self._quiet:
            os.environ["TQDM_DISABLE"] = "1"
            suppress_token = _suppress_progress.set(True)

        try:
            extractors: list[BaseExtractor] = [
                NVDExtractor(url, timeout=timeout) for url in self._urls
            ]
            transformers: list[BaseTransform] = []

            if cvss:
                from vulnflow.transforms.flatten import CVSSFlattenTransform

                transformers.append(CVSSFlattenTransform())

            if epss:
                from vulnflow.transforms.enrichers.epss import EPSSEnricher
                from vulnflow.transforms.flatten import EPSSFlattenTransform

                transformers.extend([EPSSEnricher(), EPSSFlattenTransform()])

            if kev:
                from vulnflow.transforms.enrichers.kev import KEVEnricher
                from vulnflow.transforms.flatten import KEVFlattenTransform

                transformers.extend([KEVEnricher(), KEVFlattenTransform()])

            pipeline = VulnPipeline(
                extractors=extractors,
                transformers=transformers,
            )

            result = pipeline.run(
                quiet=self._quiet,
                extract_desc="Extracting NVD Data",
            )
            dataset = result.dataset

            if dataset is None:
                raise RuntimeError(
                    f"All {len(self._urls)} feeds failed; no data extracted"
                )

            logger.info("Library built: %d records", len(dataset))
            return dataset

        finally:
            # Restore progress suppression state
            if self._quiet:
                if suppress_token is not None:
                    _suppress_progress.reset(suppress_token)
                if old_tqdm_disable is None:
                    os.environ.pop("TQDM_DISABLE", None)
                else:
                    os.environ["TQDM_DISABLE"] = old_tqdm_disable

    @classmethod
    def feed_url(cls, feed: int | str) -> str:
        """Return NVD feed URL for a year or special feed.

        Args:
            feed: Year (e.g., 2024) or feed name ("modified", "recent").

        Returns:
            Full URL to NVD JSON feed (.json.gz).

        Example:
            >>> CacheBuilder.feed_url(2024)
            'https://nvd.nist.gov/feeds/json/cve/2.0/nvdcve-2.0-2024.json.gz'
            >>> CacheBuilder.feed_url("modified")
            'https://nvd.nist.gov/feeds/json/cve/2.0/nvdcve-2.0-Modified.json.gz'
        """
        if isinstance(feed, int):
            return f"{cls.NVD_BASE_URL}/nvdcve-2.0-{feed}.json.gz"
        name = feed.capitalize()
        return f"{cls.NVD_BASE_URL}/nvdcve-2.0-{name}.json.gz"
