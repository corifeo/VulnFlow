"""
EPSS Enricher - FIRST EPSS probability scores

Provides:
- EPSSEnricher: Enriches records with EPSS data via CSV join

Layer: Transform/Enrichers
Dependencies: dataset, utils.http
Used by: dataset.transform(), pipeline
"""

from __future__ import annotations

import csv
import io
import logging
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
import numpy as np

if TYPE_CHECKING:
    from vulnflow.dataset import VulnDataset

from vulnflow.registry import FieldDefinition, register_transform
from vulnflow.transforms.base import BaseEnricher
from vulnflow.utils.http import HttpFetcher, is_url

logger = logging.getLogger(__name__)

DEFAULT_EPSS_URL = "https://epss.empiricalsecurity.com/epss_scores-current.csv.gz"


@register_transform(default=True)
class EPSSEnricher(BaseEnricher):
    """Enrich dataset with FIRST EPSS probability scores.

    BulkEnricher pattern: Downloads full EPSS CSV, caches locally,
    joins on CVE ID using vectorized operations.
    """

    name: str = "EPSSEnricher"
    depends_on: list[str] = []

    OUTPUT_SCHEMA: list[FieldDefinition] = [
        FieldDefinition(
            "epss_probability", "float", "EPSS exploit probability", "EPSSEnricher"
        ),
        FieldDefinition(
            "epss_percentile", "float", "EPSS percentile rank", "EPSSEnricher"
        ),
        FieldDefinition("epss_date", "datetime", "EPSS score date", "EPSSEnricher"),
        FieldDefinition("epss_model", "int", "EPSS model version", "EPSSEnricher"),
    ]

    def __init__(
        self,
        source: str | Path | None = None,
        cache_ttl: int = 86400,
        cache_dir: Path | str = ".cache",
        timeout: float = 120.0,
    ) -> None:
        """Initialize enricher with EPSS source.

        Args:
            source: Path to EPSS CSV file, URL, or None for default EPSS URL.
            cache_ttl: Cache time-to-live in seconds (default 24h)
            cache_dir: Directory for HTTP response caching
            timeout: HTTP request timeout in seconds (longer for large EPSS file)
        """
        super().__init__()

        # Determine source type
        if source is None:
            self.source: str | Path = DEFAULT_EPSS_URL
        elif isinstance(source, str) and is_url(source):
            self.source = source
        else:
            self.source = Path(source)

        self.cache_ttl = cache_ttl
        self.cache_dir = Path(cache_dir)
        self.timeout = timeout
        self._cache: dict[str, dict[str, Any]] | None = None
        self._cache_time: float = 0.0
        self._http_fetcher: HttpFetcher | None = None

    async def transform(self, dataset: VulnDataset) -> None:  # type: ignore[override]
        """Enrich dataset with EPSS data (fully vectorized).

        Args:
            dataset: VulnDataset to enrich (modifies in place)
        """
        # Load EPSS data with caching
        epss_data = await self._get_cached_data_async()

        # Get CVE IDs from dataset
        cve_ids = dataset.get_column("cve_id")
        n = len(dataset)

        # Build parallel arrays from EPSS dict
        epss_cve_ids = np.array(list(epss_data.keys()), dtype=object)
        epss_probs = np.array(
            [epss_data[k]["probability"] for k in epss_cve_ids], dtype=np.float64
        )
        epss_percs = np.array(
            [epss_data[k]["percentile"] for k in epss_cve_ids], dtype=np.float64
        )
        epss_dates = np.array(
            [epss_data[k].get("date") for k in epss_cve_ids], dtype=object
        )
        epss_models = np.array(
            [epss_data[k].get("model", 3) for k in epss_cve_ids], dtype=object
        )

        # Vectorized matching using np.searchsorted
        sort_idx = np.argsort(epss_cve_ids)
        epss_cve_ids_sorted = epss_cve_ids[sort_idx]
        indices = np.searchsorted(epss_cve_ids_sorted, cve_ids)
        valid = (indices < len(epss_cve_ids_sorted)) & (
            epss_cve_ids_sorted[indices] == cve_ids
        )

        # Initialize output arrays
        epss_probability = np.full(n, np.nan, dtype=np.float64)
        epss_percentile = np.full(n, np.nan, dtype=np.float64)
        epss_date = np.full(n, None, dtype=object)
        epss_model = np.full(n, None, dtype=object)

        # Assign using boolean indexing
        epss_probability[valid] = epss_probs[sort_idx[indices[valid]]]
        epss_percentile[valid] = epss_percs[sort_idx[indices[valid]]]
        epss_date[valid] = epss_dates[sort_idx[indices[valid]]]
        epss_model[valid] = epss_models[sort_idx[indices[valid]]]

        # Set columns
        dataset.set_column("epss_probability", epss_probability)
        dataset.set_column("epss_percentile", epss_percentile)
        dataset.set_column("epss_date", epss_date)
        dataset.set_column("epss_model", epss_model)

        # Update sources
        matched_indices = np.where(valid)[0].tolist()
        if matched_indices:
            dataset.update_sources(matched_indices, "EPSS")

    async def _get_cached_data_async(self) -> dict[str, dict[str, Any]]:
        """Get EPSS data from cache or load fresh (async)."""
        now = time.time()
        if self._cache is None or now - self._cache_time > self.cache_ttl:
            self._cache = await self._load_epss_data()
            self._cache_time = now
        return self._cache

    async def _load_epss_data(self) -> dict[str, dict[str, Any]]:
        """Load EPSS data from file or URL.

        Returns:
            Dict mapping CVE ID to EPSS entry dict
        """
        if isinstance(self.source, Path):
            return self._load_epss_from_file(self.source)
        else:
            return await self._load_epss_from_url(self.source)

    def _load_epss_from_file(self, path: Path) -> dict[str, dict[str, Any]]:
        """Load EPSS data from local file.

        Args:
            path: Path to CSV file

        Returns:
            Dict mapping CVE ID to EPSS entry dict
        """
        try:
            content = path.read_text(encoding="utf-8")
            return self._parse_epss_csv(content)
        except Exception as e:
            logger.warning("Failed to load EPSS data from %s: %s", path, e)
            return {}

    async def _load_epss_from_url(self, url: str) -> dict[str, dict[str, Any]]:
        """Load EPSS data from URL.

        Args:
            url: HTTP/HTTPS URL to fetch

        Returns:
            Dict mapping CVE ID to EPSS entry dict
        """
        if self._http_fetcher is None:
            self._http_fetcher = HttpFetcher(
                cache_dir=self.cache_dir / "epss",
                timeout=self.timeout,
            )

        try:
            # HttpFetcher auto-decompresses gzip
            content, _ = await self._http_fetcher.fetch(url, self.cache_ttl)
            return self._parse_epss_csv(content.decode("utf-8"))

        except httpx.HTTPError as e:
            logger.warning("Failed to fetch EPSS data from %s: %s", url, e)
            return {}
        except Exception as e:
            logger.warning("Failed to load EPSS data from %s: %s", url, e)
            return {}

    def _parse_epss_csv(self, content: str) -> dict[str, dict[str, Any]]:
        """Parse EPSS data from CSV content.

        Args:
            content: CSV content as string (may include # comment header)

        Returns:
            Dict mapping CVE ID to EPSS entry dict
        """
        epss_data: dict[str, dict[str, Any]] = {}

        # Split content into lines
        all_lines = content.splitlines(keepends=True)

        # Extract date from header comment
        # Format: #model_version:vX,score_date:YYYY-MM-DDTHH:MM:SSZ
        score_date: datetime | None = None
        header_lines = [line for line in all_lines if line.startswith("#")]
        if header_lines:
            date_match = re.search(r"score_date:(\d{4}-\d{2}-\d{2})", header_lines[0])
            if date_match:
                score_date = datetime.strptime(date_match.group(1), "%Y-%m-%d")
                score_date = score_date.replace(tzinfo=UTC)

        # Filter out comment lines before CSV parsing
        data_lines = [line for line in all_lines if not line.startswith("#")]
        reader = csv.DictReader(io.StringIO("".join(data_lines)))

        for row in reader:
            cve_id = row["cve"]

            epss_data[cve_id] = {
                "probability": float(row["epss"]),
                "percentile": float(row["percentile"]),
                "date": score_date,  # Use date from header
                "model": 3,  # Current EPSS model version
            }

        return epss_data
