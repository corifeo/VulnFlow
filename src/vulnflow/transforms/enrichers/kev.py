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


@register_enricher(default=True)
class KEVEnricher:
    """Enrich dataset with CISA KEV catalog data.

    BulkEnricher pattern: Downloads full KEV CSV, caches locally,
    joins on CVE ID using vectorized operations.
    """

    name: str = "KEVEnricher"
    depends_on: list[str] = []

    SCHEMA_VERSION: str = "1.0.0"
    OUTPUT_SCHEMA: list[dict[str, str]] = [
        {"name": "kev_date_added", "type": "datetime", "description": "Date added to KEV catalog"},
        {"name": "kev_due_date", "type": "datetime", "description": "Remediation due date"},
        {"name": "kev_required_action", "type": "str", "description": "Required remediation action"},
        {"name": "kev_vendor", "type": "str", "description": "Vendor name from KEV"},
        {"name": "kev_product", "type": "str", "description": "Product name from KEV"},
        {"name": "kev_ransomware", "type": "str", "description": "Known ransomware campaign use"},
    ]

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

    async def transform(self, dataset: VulnDataset) -> None:
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

        # Use vectorized operations to find matches
        # Build lookup array from KEV catalog
        kev_cve_ids = np.array(list(kev_data.keys()), dtype=object)

        # Find matches using numpy indexing (vectorized)
        matches_mask = np.isin(cve_ids, kev_cve_ids)
        matched_indices = np.where(matches_mask)[0].tolist()

        # For matched records, populate using direct indexing
        for idx in matched_indices:
            cve_id = cve_ids[idx]
            if cve_id in kev_data:
                entry = kev_data[cve_id]
                kev_date_added[idx] = entry["date_added"]
                kev_due_date[idx] = entry["due_date"]
                kev_required_action[idx] = entry["required_action"]
                kev_vendor[idx] = entry["vendor"]
                kev_product[idx] = entry["product"]
                kev_ransomware[idx] = entry["ransomware"]

        # Add columns to dataset
        dataset.set_column("kev_date_added", np.array(kev_date_added, dtype=object))
        dataset.set_column("kev_due_date", np.array(kev_due_date, dtype=object))
        dataset.set_column(
            "kev_required_action", np.array(kev_required_action, dtype=object)
        )
        dataset.set_column("kev_vendor", np.array(kev_vendor, dtype=object))
        dataset.set_column("kev_product", np.array(kev_product, dtype=object))
        dataset.set_column("kev_ransomware", np.array(kev_ransomware, dtype=object))

        # Update sources for matched records
        if matched_indices:
            dataset.update_sources(matched_indices, "KEV")

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
