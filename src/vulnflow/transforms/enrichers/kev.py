"""
KEV Enricher - CISA Known Exploited Vulnerabilities catalog

Provides:
- KEVEnricher: Enriches records with KEV data via CSV join

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
class KEVEnricher:
    """Enrich dataset with CISA KEV catalog data.

    BulkEnricher pattern: Downloads full KEV CSV, caches locally,
    joins on CVE ID using vectorized operations.
    """

    name: str = "KEVEnricher"

    def __init__(
        self,
        source: str | Path | None = None,
        cache_ttl: int = 86400,
    ) -> None:
        """Initialize enricher with KEV source.

        Args:
            source: Path to KEV CSV file. If None, will fetch from CISA API (future).
            cache_ttl: Cache time-to-live in seconds (default 24h)
        """
        self.source = Path(source) if source else None
        self.cache_ttl = cache_ttl
        self._cache: dict[str, dict[str, Any]] | None = None
        self._cache_time: float = 0.0

    async def enrich(self, dataset: VulnDataset) -> None:
        """Add KEV columns to dataset.

        Args:
            dataset: VulnDataset to enrich (modifies in place)
        """
        # Load KEV catalog with caching
        kev_data = self._get_cached_data()

        # Get CVE IDs from dataset
        cve_ids = dataset.get_column("cve_id")
        n = len(dataset)

        # Initialize KEV columns with None
        kev_date_added: list[datetime | None] = [None] * n
        kev_due_date: list[datetime | None] = [None] * n
        kev_required_action: list[str | None] = [None] * n
        kev_vendor: list[str | None] = [None] * n
        kev_product: list[str | None] = [None] * n
        kev_ransomware: list[str | None] = [None] * n

        # Match by CVE ID
        for i, cve_id in enumerate(cve_ids):
            if cve_id in kev_data:
                entry = kev_data[cve_id]
                kev_date_added[i] = entry["date_added"]
                kev_due_date[i] = entry["due_date"]
                kev_required_action[i] = entry["required_action"]
                kev_vendor[i] = entry["vendor"]
                kev_product[i] = entry["product"]
                kev_ransomware[i] = entry["ransomware"]

        # Add columns to dataset
        dataset.set_column("kev_date_added", np.array(kev_date_added, dtype=object))
        dataset.set_column("kev_due_date", np.array(kev_due_date, dtype=object))
        dataset.set_column(
            "kev_required_action", np.array(kev_required_action, dtype=object)
        )
        dataset.set_column("kev_vendor", np.array(kev_vendor, dtype=object))
        dataset.set_column("kev_product", np.array(kev_product, dtype=object))
        dataset.set_column("kev_ransomware", np.array(kev_ransomware, dtype=object))

    def _get_cached_data(self) -> dict[str, dict[str, Any]]:
        """Get KEV data from cache or load fresh."""
        now = time.time()
        if self._cache is None or now - self._cache_time > self.cache_ttl:
            self._cache = self._load_kev_csv()
            self._cache_time = now
        return self._cache

    def _load_kev_csv(self) -> dict[str, dict[str, Any]]:
        """Load KEV catalog from CSV file.

        Returns:
            Dict mapping CVE ID to KEV entry dict
        """
        if self.source is None:
            raise ValueError("KEV source not specified")

        kev_data: dict[str, dict[str, Any]] = {}

        try:
            with open(self.source, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cve_id = row["cveID"]

                    # Parse dates
                    date_added = datetime.strptime(
                        row["dateAdded"], "%Y-%m-%d"
                    ).replace(tzinfo=UTC)
                    due_date = datetime.strptime(row["dueDate"], "%Y-%m-%d").replace(
                        tzinfo=UTC
                    )

                    kev_data[cve_id] = {
                        "date_added": date_added,
                        "due_date": due_date,
                        "required_action": row.get("requiredAction"),
                        "vendor": row.get("vendorProject"),
                        "product": row.get("product"),
                        "ransomware": row.get("knownRansomwareCampaignUse"),
                    }
        except Exception as e:
            logger.warning("Failed to load KEV data from %s: %s", self.source, e)
            return {}

        return kev_data
