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
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vulnflow.dataset import VulnDataset
    from vulnflow.extractors.base import BaseExtractor
    from vulnflow.transforms.base import BaseTransform

logger = logging.getLogger(__name__)

NVD_BASE_URL = "https://nvd.nist.gov/feeds/json/cve/2.0"


def _parse_duration(value: str | timedelta) -> timedelta:
    """Parse duration from string ('7d', '24h') or pass through timedelta."""
    if isinstance(value, timedelta):
        return value
    if not isinstance(value, str):
        msg = f"Cannot parse duration from {type(value).__name__}"
        raise ValueError(msg)
    stripped = value.strip().lower()
    try:
        if stripped.endswith("d"):
            return timedelta(days=float(stripped[:-1]))
        if stripped.endswith("h"):
            return timedelta(hours=float(stripped[:-1]))
    except ValueError:
        pass
    msg = f"Cannot parse duration: '{value}'. Use '7d' or '24h' format."
    raise ValueError(msg)


def _read_parquet_provenance(path: Path) -> dict[str, Any] | None:
    """Read provenance metadata from parquet file without loading data.

    Uses pq.read_schema() for minimal I/O (reads only footer metadata).
    Returns None if file doesn't exist or has no provenance.
    """
    if not path.exists():
        return None
    import pyarrow.parquet as pq

    schema = pq.read_schema(path)
    meta = schema.metadata or {}
    prov_bytes = meta.get(b"vulnflow:provenance")
    if not prov_bytes:
        return None
    from vulnflow.utils import json as vjson

    return vjson.loads(prov_bytes)  # type: ignore[no-any-return]


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
        cpe: bool = True,
        cwe: bool = True,
        capec: bool = False,
        stride: bool = False,
    ) -> VulnDataset:
        """Build vulnerability dataset with concurrent extraction.

        Uses ThreadPoolExecutor for concurrent feed extraction (no subprocess
        overhead). Enriches with EPSS/KEV/CWE by default (bulk downloads,
        cached locally).

        Args:
            timeout: HTTP timeout in seconds per feed.
            cvss: Flatten CVSS scores to scalar columns (default: True).
            epss: Enrich with EPSS exploitation probabilities (default: True).
            kev: Enrich with CISA KEV catalog data (default: True).
            cpe: Flatten CPE vendor/product/version columns (default: True).
            cwe: Enrich with CWE weakness names and categories (default: True).
            capec: Enrich with CAPEC attack patterns (default: False).
            stride: Infer STRIDE threat categories (default: False).

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

            if cpe:
                from vulnflow.transforms.flatten import CPEFlattenTransform

                transformers.append(CPEFlattenTransform())

            if cwe:
                from vulnflow.transforms.enrichers.cwe import CWEEnricher
                from vulnflow.transforms.flatten import CWEFlattenTransform

                transformers.extend([CWEEnricher(), CWEFlattenTransform()])

            if capec:
                from vulnflow.transforms.enrichers.capec import CAPECEnricher
                from vulnflow.transforms.flatten import CAPECFlattenTransform

                # Auto-enable CWE enricher if not already included
                if not cwe:
                    from vulnflow.transforms.enrichers.cwe import CWEEnricher

                    transformers.append(CWEEnricher())

                transformers.extend([CAPECEnricher(), CAPECFlattenTransform()])

            if stride:
                from vulnflow.transforms.flatten.stride import STRIDEFlattenTransform
                from vulnflow.transforms.inferences.stride import STRIDEInference

                # Auto-enable CAPEC chain if not already included
                if not capec:
                    from vulnflow.transforms.enrichers.capec import CAPECEnricher
                    from vulnflow.transforms.flatten import CAPECFlattenTransform

                    if not cwe:
                        from vulnflow.transforms.enrichers.cwe import CWEEnricher

                        transformers.append(CWEEnricher())

                    transformers.extend([CAPECEnricher(), CAPECFlattenTransform()])

                transformers.extend([STRIDEInference(), STRIDEFlattenTransform()])

            pipeline = VulnPipeline(
                extractors=extractors,
                transformers=transformers,
            )

            result = pipeline.run(
                quiet=self._quiet,
                extract_desc="Processing NVD feeds",
                transform_desc="Enriching & transforming",
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

    def ensure(
        self,
        path: str | Path,
        *,
        max_age: str | timedelta | None = None,
        through: str | None = None,
        recent: bool = False,
        timeout: float = 120.0,
        **build_kwargs: Any,
    ) -> VulnDataset:
        """Ensure cache file meets freshness requirements.

        Checks existing parquet file and updates or rebuilds as needed.

        Args:
            path: Path to parquet cache file.
            max_age: Maximum cache age ('7d', '24h', or timedelta).
                Rebuilds if exceeded.
            through: ISO date string. Rebuilds if date_max < through.
            recent: If True, apply modified feed delta on top of cache.
            timeout: HTTP timeout per feed.
            **build_kwargs: Passed to build() (cvss, epss, kev, etc.)
        """
        from vulnflow.dataset import VulnDataset
        from vulnflow.loaders.files import ParquetLoader

        path = Path(path)

        # 1. File missing → full build
        if not path.exists():
            logger.info("Cache file missing, building: %s", path)
            ds = self.build(timeout=timeout, **build_kwargs)
            ParquetLoader(path).load(ds)
            return ds

        # 2. Read provenance (lightweight, no data load)
        prov = _read_parquet_provenance(path)

        # 3. Determine if rebuild is needed
        needs_rebuild = False

        if max_age is not None and prov is not None:
            updated_at = prov.get("updated_at")
            if updated_at is not None:
                from datetime import UTC

                cache_time = datetime.fromisoformat(updated_at)
                age_limit = datetime.now(UTC) - _parse_duration(max_age)
                if cache_time < age_limit:
                    logger.info("Cache stale (max_age), rebuilding")
                    needs_rebuild = True
            else:
                needs_rebuild = True

        if not needs_rebuild and through is not None:
            date_max = prov.get("date_max") if prov else None
            if date_max is None:
                logger.info("Cache missing date_max, rebuilding for through=%s", through)
                needs_rebuild = True
            elif date_max < through:
                logger.info(
                    "Cache date_max=%s < through=%s, rebuilding", date_max, through
                )
                needs_rebuild = True

        # 4. Fresh and no recent → return cached
        if not needs_rebuild and not recent:
            logger.info("Cache fresh, loading: %s", path)
            return VulnDataset.from_parquet(str(path))

        # 5. Rebuild: full build, save, return
        logger.info("Rebuilding cache: %s", path)
        ds = self.build(timeout=timeout, **build_kwargs)
        ParquetLoader(path).load(ds)
        return ds

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
