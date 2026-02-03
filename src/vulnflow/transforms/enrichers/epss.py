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

from vulnflow.registry import FieldDefinition, register_enricher
from vulnflow.transforms.base import BaseEnricher

logger = logging.getLogger(__name__)


@register_enricher(default=True)
class EPSSEnricher(BaseEnricher):
    """Enrich dataset with FIRST EPSS probability scores.

    BulkEnricher pattern: Downloads full EPSS CSV, caches locally,
    joins on CVE ID using vectorized operations.
    """

    name: str = "EPSSEnricher"
    depends_on: list[str] = []

    OUTPUT_SCHEMA: list[FieldDefinition] = [
        FieldDefinition("epss_probability", "float", "EPSS exploit probability", "EPSSEnricher"),
        FieldDefinition("epss_percentile", "float", "EPSS percentile rank", "EPSSEnricher"),
        FieldDefinition("epss_date", "datetime", "EPSS score date", "EPSSEnricher"),
        FieldDefinition("epss_model", "int", "EPSS model version", "EPSSEnricher"),
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
        super().__init__()

        self.source = Path(source) if source else None
        self.cache_ttl = cache_ttl
        self._cache: dict[str, dict[str, Any]] | None = None
        self._cache_time: float = 0.0

    async def transform(self, dataset: VulnDataset) -> None:  # type: ignore[override]
        """Enrich dataset with EPSS data (fully vectorized).

        Args:
            dataset: VulnDataset to enrich (modifies in place)
        """
        # Load EPSS data with caching
        epss_data = self._get_cached_data()

        # Get CVE IDs from dataset
        cve_ids = dataset.get_column("cve_id")
        n = len(dataset)

        # Build parallel arrays from EPSS dict
        epss_cve_ids = np.array(list(epss_data.keys()), dtype=object)
        epss_probs = np.array([epss_data[k]["probability"] for k in epss_cve_ids], dtype=np.float64)
        epss_percs = np.array([epss_data[k]["percentile"] for k in epss_cve_ids], dtype=np.float64)
        epss_dates = np.array([epss_data[k].get("date") for k in epss_cve_ids], dtype=object)
        epss_models = np.array([epss_data[k].get("model", 3) for k in epss_cve_ids], dtype=object)

        # Vectorized matching using np.searchsorted
        sort_idx = np.argsort(epss_cve_ids)
        epss_cve_ids_sorted = epss_cve_ids[sort_idx]
        indices = np.searchsorted(epss_cve_ids_sorted, cve_ids)
        valid = (indices < len(epss_cve_ids_sorted)) & (epss_cve_ids_sorted[indices] == cve_ids)

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
                # Read all lines
                all_lines = f.readlines()

                # Extract date from header comment
                # Format: #model_version:vX,score_date:YYYY-MM-DDTHH:MM:SSZ
                score_date: datetime | None = None
                header_lines = [line for line in all_lines if line.startswith('#')]
                if header_lines:
                    import re
                    date_match = re.search(r'score_date:(\d{4}-\d{2}-\d{2})', header_lines[0])
                    if date_match:
                        score_date = datetime.strptime(date_match.group(1), "%Y-%m-%d")
                        score_date = score_date.replace(tzinfo=UTC)

                # Filter out comment lines before CSV parsing
                data_lines = [line for line in all_lines if not line.startswith('#')]
                reader = csv.DictReader(data_lines)

                for row in reader:
                    cve_id = row["cve"]

                    epss_data[cve_id] = {
                        "probability": float(row["epss"]),
                        "percentile": float(row["percentile"]),
                        "date": score_date,  # Use date from header
                        "model": 3,  # Current EPSS model version
                    }
        except Exception as e:
            logger.warning("Failed to load EPSS data from %s: %s", self.source, e)
            return {}

        return epss_data
