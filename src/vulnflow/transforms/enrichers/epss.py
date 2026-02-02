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
from datetime import UTC, datetime
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
    depends_on: list[str] = []

    SCHEMA_VERSION: str = "1.0.0"
    OUTPUT_SCHEMA: list[dict[str, str]] = [
        {"name": "epss_probability", "type": "float", "description": "EPSS probability score"},
        {"name": "epss_percentile", "type": "float", "description": "EPSS percentile rank"},
        {"name": "epss_date", "type": "datetime", "description": "Score calculation date"},
        {"name": "epss_model", "type": "int", "description": "EPSS model version (1, 2, or 3)"},
    ]

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

    async def transform(self, dataset: VulnDataset) -> None:
        """Add EPSS columns to dataset.

        Args:
            dataset: VulnDataset to enrich (modifies in place)
        """
        # Load EPSS data with caching
        epss_data = self._get_cached_data()

        # Get CVE IDs from dataset
        cve_ids = dataset.get_column("cve_id")
        n = len(dataset)

        # Initialize EPSS columns
        epss_probability: np.ndarray = np.full(n, np.nan, dtype=np.float64)
        epss_percentile: np.ndarray = np.full(n, np.nan, dtype=np.float64)
        epss_date: np.ndarray = np.array([None] * n, dtype=object)
        epss_model: np.ndarray = np.array([None] * n, dtype=object)

        # Track matched indices for source tracking
        matched_indices: list[int] = []

        # Match by CVE ID
        for i, cve_id in enumerate(cve_ids):
            if cve_id in epss_data:
                matched_indices.append(i)
                entry = epss_data[cve_id]
                epss_probability[i] = entry["probability"]
                epss_percentile[i] = entry["percentile"]
                epss_date[i] = entry.get("date")
                epss_model[i] = entry.get("model", 3)  # Default to model 3

        # Add columns to dataset
        dataset.set_column("epss_probability", epss_probability)
        dataset.set_column("epss_percentile", epss_percentile)
        dataset.set_column("epss_date", epss_date)
        dataset.set_column("epss_model", epss_model)

        # Update sources for matched records
        if matched_indices:
            dataset.update_sources(matched_indices, "EPSS")

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

                    # Parse date if present
                    date_str = row.get("date")
                    parsed_date: datetime | None = None
                    if date_str:
                        parsed_date = datetime.strptime(date_str, "%Y-%m-%d")
                        parsed_date = parsed_date.replace(tzinfo=UTC)

                    epss_data[cve_id] = {
                        "probability": float(row["epss"]),
                        "percentile": float(row["percentile"]),
                        "date": parsed_date,
                        "model": 3,  # Current EPSS model version
                    }
        except Exception as e:
            logger.warning("Failed to load EPSS data from %s: %s", self.source, e)
            return {}

        return epss_data
