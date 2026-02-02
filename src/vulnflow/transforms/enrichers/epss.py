"""
EPSS Enricher - FIRST EPSS probability scores

Provides:
- EPSSEnricher: Enriches records with EPSS data via CSV join

Layer: Transform/Enrichers
Dependencies: dataset
Used by: dataset.transform(), pipeline
"""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from vulnflow.dataset import VulnDataset

from vulnflow.registry import register_enricher

logger = logging.getLogger(__name__)


@register_enricher
class EPSSEnricher:
    """Enrich dataset with FIRST EPSS probability scores.

    BulkEnricher pattern: Downloads full EPSS CSV, caches locally,
    joins on CVE ID using vectorized operations.
    """

    name: str = "EPSSEnricher"

    def __init__(
        self,
        source: str | Path | None = None,
        cache_ttl: int = 86400,
    ) -> None:
        """Initialize enricher with EPSS source.

        Args:
            source: Path to EPSS CSV file. If None, will fetch from FIRST API (future).
            cache_ttl: Cache time-to-live in seconds (default 24h)
        """
        self.source = Path(source) if source else None
        self.cache_ttl = cache_ttl
        self._cache: dict[str, dict[str, Any]] | None = None
        self._cache_time: float = 0.0

    async def enrich(self, dataset: VulnDataset) -> None:
        """Add EPSS columns to dataset.

        Args:
            dataset: VulnDataset to enrich (modifies in place)
        """
        # Load EPSS data with caching
        epss_data = self._get_cached_data()

        # Get CVE IDs from dataset
        cve_ids = dataset.get_column("cve_id")
        n = len(dataset)

        # Initialize EPSS columns with NaN
        epss_probability: np.ndarray = np.full(n, np.nan, dtype=np.float64)
        epss_percentile: np.ndarray = np.full(n, np.nan, dtype=np.float64)

        # Match by CVE ID
        for i, cve_id in enumerate(cve_ids):
            if cve_id in epss_data:
                entry = epss_data[cve_id]
                epss_probability[i] = entry["probability"]
                epss_percentile[i] = entry["percentile"]

        # Add columns to dataset
        dataset.set_column("epss_probability", epss_probability)
        dataset.set_column("epss_percentile", epss_percentile)

    def _get_cached_data(self) -> dict[str, dict[str, Any]]:
        """Get EPSS data from cache or load fresh."""
        now = time.time()
        if self._cache is None or now - self._cache_time > self.cache_ttl:
            self._cache = self._load_epss_csv()
            self._cache_time = now
        return self._cache

    def _load_epss_csv(self) -> dict[str, dict[str, Any]]:
        """Load EPSS data from CSV file.

        Returns:
            Dict mapping CVE ID to EPSS entry dict
        """
        if self.source is None:
            raise ValueError("EPSS source not specified")

        epss_data: dict[str, dict[str, Any]] = {}

        try:
            with open(self.source, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cve_id = row["cve"]

                    epss_data[cve_id] = {
                        "probability": float(row["epss"]),
                        "percentile": float(row["percentile"]),
                    }
        except Exception as e:
            logger.warning("Failed to load EPSS data from %s: %s", self.source, e)
            return {}

        return epss_data
