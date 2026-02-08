"""
CacheBuilder - Commodity class for building NVD vulnerability libraries

Wraps VulnPipeline with NVD-specific URL knowledge for one-line library
construction. Supports multiple modes: all years, specific years, modified
feed only, recent feed only.

Layer: User-facing helpers
Dependencies: pipeline, extractors.nvd, loaders.files
Used by: Notebooks, user scripts
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vulnflow.dataset import VulnDataset

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
        >>> ds = CacheBuilder(years=[2024, 2025]).build(workers=2)
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
        workers: int = 4,
        timeout: float = 120.0,
        cvss: bool = True,
    ) -> VulnDataset:
        """Build vulnerability dataset with parallel extraction.

        Args:
            workers: Number of parallel worker processes.
            timeout: HTTP timeout in seconds per feed.
            cvss: Apply CVSSFlattenTransform to extract scalar CVSS columns (default: True).

        Returns:
            Merged VulnDataset with all extracted records.

        Raises:
            RuntimeError: If all feeds fail and no parquet files are produced.
        """
        from vulnflow.dataset import VulnDataset
        from vulnflow.extractors.nvd import NVDExtractor
        from vulnflow.loaders.files import ParquetLoader
        from vulnflow.pipeline import VulnPipeline
        from vulnflow.utils.progress import _suppress_progress

        logger.info(
            "Building library from %d feeds with %d workers",
            len(self._urls),
            workers,
        )

        # Suppress progress output when quiet mode is enabled
        old_tqdm_disable = os.environ.get("TQDM_DISABLE")
        suppress_token = None
        if self._quiet:
            os.environ["TQDM_DISABLE"] = "1"
            suppress_token = _suppress_progress.set(True)

        temp_dir = Path(tempfile.mkdtemp(prefix="vulnflow_cache_"))
        try:
            # Parallel extract -> temp parquets
            temp_paths = [
                temp_dir / f"part_{i}.parquet" for i in range(len(self._urls))
            ]

            pipeline = VulnPipeline(
                extractors=[NVDExtractor(url, timeout=timeout) for url in self._urls],
                loaders=[ParquetLoader(p) for p in temp_paths],
            )

            pipeline.run(parallel=True, workers=workers)

            # Check which temp files actually exist (some workers may have failed)
            existing_paths = [p for p in temp_paths if p.exists()]
            missing_paths = [p for p in temp_paths if not p.exists()]

            if missing_paths:
                failed_feeds = [
                    self._urls[i]
                    for i, p in enumerate(temp_paths)
                    if not p.exists()
                ]
                logger.warning(
                    "%d feed(s) produced no output: %s",
                    len(missing_paths),
                    ", ".join(failed_feeds),
                )

            if not existing_paths:
                raise RuntimeError(
                    f"All {len(temp_paths)} feeds failed; no data to merge"
                )

            # Merge temp parquets (only existing files)
            logger.info(
                "Merging %d of %d parquet files",
                len(existing_paths),
                len(temp_paths),
            )
            merged = VulnDataset.from_parquet([str(p) for p in existing_paths])

            # Apply CVSS flatten transform by default
            if cvss:
                from vulnflow.transforms.flatten import CVSSFlattenTransform

                logger.info("Applying CVSSFlattenTransform")
                transform = CVSSFlattenTransform()
                transform.transform(merged)

            logger.info("Library built: %d records", len(merged))
            return merged

        finally:
            # Restore progress suppression state
            if self._quiet:
                if suppress_token is not None:
                    _suppress_progress.reset(suppress_token)
                if old_tqdm_disable is None:
                    os.environ.pop("TQDM_DISABLE", None)
                else:
                    os.environ["TQDM_DISABLE"] = old_tqdm_disable
            shutil.rmtree(temp_dir, ignore_errors=True)

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
