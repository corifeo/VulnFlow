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

from vulnflow.registry import FieldDefinition, register_enricher
from vulnflow.transforms.base import BaseEnricher

logger = logging.getLogger(__name__)


@register_enricher(default=True)
class KEVEnricher(BaseEnricher):
    """Enrich dataset with CISA KEV catalog data.

    BulkEnricher pattern: Downloads full KEV CSV, caches locally,
    joins on CVE ID using vectorized operations.
    """

    name: str = "KEVEnricher"
    depends_on: list[str] = []

    OUTPUT_SCHEMA: list[FieldDefinition] = [
        FieldDefinition("kev_date_added", "datetime", "Date added to KEV catalog", "KEVEnricher"),
        FieldDefinition("kev_due_date", "datetime", "Remediation due date", "KEVEnricher"),
        FieldDefinition("kev_required_action", "str", "Required remediation action", "KEVEnricher"),
        FieldDefinition("kev_vendor", "str", "Vendor name from KEV", "KEVEnricher"),
        FieldDefinition("kev_product", "str", "Product name from KEV", "KEVEnricher"),
        FieldDefinition("kev_ransomware", "str", "Known ransomware campaign use", "KEVEnricher"),
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
        super().__init__()

        self.source = Path(source) if source else None
        self.cache_ttl = cache_ttl
        self._cache: dict[str, dict[str, Any]] | None = None
        self._cache_time: float = 0.0

    async def transform(self, dataset: VulnDataset) -> None:  # type: ignore[override]
        """Enrich dataset with KEV data (fully vectorized).

        Args:
            dataset: VulnDataset to enrich (modifies in place)
        """
        # Load KEV catalog with caching
        kev_data = self._get_cached_data()

        # Get CVE IDs from dataset
        cve_ids = dataset.get_column("cve_id")
        n = len(dataset)

        # Build parallel arrays from KEV dict
        kev_cve_ids = np.array(list(kev_data.keys()), dtype=object)
        kev_dates_added = np.array([kev_data[k]["date_added"] for k in kev_cve_ids], dtype=object)
        kev_due_dates = np.array([kev_data[k]["due_date"] for k in kev_cve_ids], dtype=object)
        kev_actions = np.array([kev_data[k]["required_action"] for k in kev_cve_ids], dtype=object)
        kev_vendors = np.array([kev_data[k]["vendor"] for k in kev_cve_ids], dtype=object)
        kev_products = np.array([kev_data[k]["product"] for k in kev_cve_ids], dtype=object)
        kev_ransomware = np.array([kev_data[k]["ransomware"] for k in kev_cve_ids], dtype=object)

        # Vectorized matching using np.searchsorted (requires sorted keys)
        sort_idx = np.argsort(kev_cve_ids)
        kev_cve_ids_sorted = kev_cve_ids[sort_idx]

        # Find position of each CVE in sorted KEV list
        indices = np.searchsorted(kev_cve_ids_sorted, cve_ids)

        # Mask for valid matches
        valid = (indices < len(kev_cve_ids_sorted)) & (kev_cve_ids_sorted[indices] == cve_ids)

        # Initialize output arrays
        kev_date_added = np.full(n, None, dtype=object)
        kev_due_date = np.full(n, None, dtype=object)
        kev_required_action = np.full(n, None, dtype=object)
        kev_vendor = np.full(n, None, dtype=object)
        kev_product = np.full(n, None, dtype=object)
        kev_ransomware_out = np.full(n, None, dtype=object)

        # Assign using boolean indexing (fully vectorized)
        kev_date_added[valid] = kev_dates_added[sort_idx[indices[valid]]]
        kev_due_date[valid] = kev_due_dates[sort_idx[indices[valid]]]
        kev_required_action[valid] = kev_actions[sort_idx[indices[valid]]]
        kev_vendor[valid] = kev_vendors[sort_idx[indices[valid]]]
        kev_product[valid] = kev_products[sort_idx[indices[valid]]]
        kev_ransomware_out[valid] = kev_ransomware[sort_idx[indices[valid]]]

        # Set columns
        dataset.set_column("kev_date_added", kev_date_added)
        dataset.set_column("kev_due_date", kev_due_date)
        dataset.set_column("kev_required_action", kev_required_action)
        dataset.set_column("kev_vendor", kev_vendor)
        dataset.set_column("kev_product", kev_product)
        dataset.set_column("kev_ransomware", kev_ransomware_out)

        # Update sources
        matched_indices = np.where(valid)[0].tolist()
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
