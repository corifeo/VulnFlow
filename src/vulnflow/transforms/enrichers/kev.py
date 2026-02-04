"""
KEV Enricher - CISA Known Exploited Vulnerabilities catalog

Provides:
- KEVEnricher: Enriches records with KEV data via CSV/JSON join

Layer: Transform/Enrichers
Dependencies: dataset, utils.http
Used by: dataset.transform(), pipeline
"""

from __future__ import annotations

import csv
import io
import logging
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
from vulnflow.utils import json
from vulnflow.utils.http import HttpFetcher, is_url

logger = logging.getLogger(__name__)

DEFAULT_KEV_URL = (
    "https://www.cisa.gov/sites/default/files/csv/known_exploited_vulnerabilities.csv"
)


@register_transform(default=True)
class KEVEnricher(BaseEnricher):
    """Enrich dataset with CISA KEV catalog data.

    BulkEnricher pattern: Downloads full KEV CSV, caches locally,
    joins on CVE ID using vectorized operations.
    """

    name: str = "KEVEnricher"
    depends_on: list[str] = []

    OUTPUT_SCHEMA: list[FieldDefinition] = [
        FieldDefinition(
            "kev_date_added", "datetime", "Date added to KEV catalog", "KEVEnricher"
        ),
        FieldDefinition(
            "kev_due_date", "datetime", "Remediation due date", "KEVEnricher"
        ),
        FieldDefinition(
            "kev_required_action", "str", "Required remediation action", "KEVEnricher"
        ),
        FieldDefinition("kev_vendor", "str", "Vendor name from KEV", "KEVEnricher"),
        FieldDefinition("kev_product", "str", "Product name from KEV", "KEVEnricher"),
        FieldDefinition(
            "kev_ransomware", "str", "Known ransomware campaign use", "KEVEnricher"
        ),
        FieldDefinition(
            "kev_vulnerability_name", "str", "Vulnerability name from KEV", "KEVEnricher"
        ),
        FieldDefinition("kev_notes", "str", "Additional notes from KEV", "KEVEnricher"),
    ]

    def __init__(
        self,
        source: str | Path | None = None,
        cache_ttl: int = 86400,
        cache_dir: Path | str = ".cache",
        timeout: float = 60.0,
    ) -> None:
        """Initialize enricher with KEV source.

        Args:
            source: Path to KEV CSV/JSON file, URL, or None for default CISA URL.
            cache_ttl: Cache time-to-live in seconds (default 24h)
            cache_dir: Directory for HTTP response caching
            timeout: HTTP request timeout in seconds
        """
        super().__init__()

        # Determine source type
        if source is None:
            self.source: str | Path = DEFAULT_KEV_URL
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
        """Enrich dataset with KEV data (fully vectorized).

        Args:
            dataset: VulnDataset to enrich (modifies in place)
        """
        # Load KEV catalog with caching
        kev_data = await self._get_cached_data_async()

        # Get CVE IDs from dataset
        cve_ids = dataset.get_column("cve_id")
        n = len(dataset)

        # Build parallel arrays from KEV dict
        kev_cve_ids = np.array(list(kev_data.keys()), dtype=object)
        kev_dates_added = np.array(
            [kev_data[k]["date_added"] for k in kev_cve_ids], dtype=object
        )
        kev_due_dates = np.array(
            [kev_data[k]["due_date"] for k in kev_cve_ids], dtype=object
        )
        kev_actions = np.array(
            [kev_data[k]["required_action"] for k in kev_cve_ids], dtype=object
        )
        kev_vendors = np.array(
            [kev_data[k]["vendor"] for k in kev_cve_ids], dtype=object
        )
        kev_products = np.array(
            [kev_data[k]["product"] for k in kev_cve_ids], dtype=object
        )
        kev_ransomware = np.array(
            [kev_data[k]["ransomware"] for k in kev_cve_ids], dtype=object
        )
        kev_vuln_names = np.array(
            [kev_data[k].get("vulnerability_name") for k in kev_cve_ids], dtype=object
        )
        kev_notes_arr = np.array(
            [kev_data[k].get("notes") for k in kev_cve_ids], dtype=object
        )

        # Vectorized matching using np.searchsorted (requires sorted keys)
        sort_idx = np.argsort(kev_cve_ids)
        kev_cve_ids_sorted = kev_cve_ids[sort_idx]

        # Find position of each CVE in sorted KEV list
        indices = np.searchsorted(kev_cve_ids_sorted, cve_ids)

        # Mask for valid matches
        valid = (indices < len(kev_cve_ids_sorted)) & (
            kev_cve_ids_sorted[indices] == cve_ids
        )

        # Initialize output arrays
        kev_date_added = np.full(n, None, dtype=object)
        kev_due_date = np.full(n, None, dtype=object)
        kev_required_action = np.full(n, None, dtype=object)
        kev_vendor = np.full(n, None, dtype=object)
        kev_product = np.full(n, None, dtype=object)
        kev_ransomware_out = np.full(n, None, dtype=object)
        kev_vulnerability_name = np.full(n, None, dtype=object)
        kev_notes = np.full(n, None, dtype=object)

        # Assign using boolean indexing (fully vectorized)
        kev_date_added[valid] = kev_dates_added[sort_idx[indices[valid]]]
        kev_due_date[valid] = kev_due_dates[sort_idx[indices[valid]]]
        kev_required_action[valid] = kev_actions[sort_idx[indices[valid]]]
        kev_vendor[valid] = kev_vendors[sort_idx[indices[valid]]]
        kev_product[valid] = kev_products[sort_idx[indices[valid]]]
        kev_ransomware_out[valid] = kev_ransomware[sort_idx[indices[valid]]]
        kev_vulnerability_name[valid] = kev_vuln_names[sort_idx[indices[valid]]]
        kev_notes[valid] = kev_notes_arr[sort_idx[indices[valid]]]

        # Set columns
        dataset.set_column("kev_date_added", kev_date_added)
        dataset.set_column("kev_due_date", kev_due_date)
        dataset.set_column("kev_required_action", kev_required_action)
        dataset.set_column("kev_vendor", kev_vendor)
        dataset.set_column("kev_product", kev_product)
        dataset.set_column("kev_ransomware", kev_ransomware_out)
        dataset.set_column("kev_vulnerability_name", kev_vulnerability_name)
        dataset.set_column("kev_notes", kev_notes)

        # Update sources
        matched_indices = np.where(valid)[0].tolist()
        if matched_indices:
            dataset.update_sources(matched_indices, "KEV")

    async def _get_cached_data_async(self) -> dict[str, dict[str, Any]]:
        """Get KEV data from cache or load fresh (async)."""
        now = time.time()
        if self._cache is None or now - self._cache_time > self.cache_ttl:
            self._cache = await self._load_kev_data()
            self._cache_time = now
        return self._cache

    async def _load_kev_data(self) -> dict[str, dict[str, Any]]:
        """Load KEV catalog from file or URL.

        Returns:
            Dict mapping CVE ID to KEV entry dict
        """
        if isinstance(self.source, Path):
            return self._load_kev_from_file(self.source)
        else:
            return await self._load_kev_from_url(self.source)

    def _load_kev_from_file(self, path: Path) -> dict[str, dict[str, Any]]:
        """Load KEV catalog from local file.

        Args:
            path: Path to CSV or JSON file

        Returns:
            Dict mapping CVE ID to KEV entry dict
        """
        try:
            content = path.read_bytes()
            if path.suffix.lower() == ".json":
                return self._parse_kev_json(content)
            else:
                return self._parse_kev_csv(content.decode("utf-8"))
        except Exception as e:
            logger.warning("Failed to load KEV data from %s: %s", path, e)
            return {}

    async def _load_kev_from_url(self, url: str) -> dict[str, dict[str, Any]]:
        """Load KEV catalog from URL.

        Args:
            url: HTTP/HTTPS URL to fetch

        Returns:
            Dict mapping CVE ID to KEV entry dict
        """
        if self._http_fetcher is None:
            self._http_fetcher = HttpFetcher(
                cache_dir=self.cache_dir / "kev",
                timeout=self.timeout,
            )

        try:
            content, fmt = await self._http_fetcher.fetch(url, self.cache_ttl)

            if fmt == "json":
                return self._parse_kev_json(content)
            else:
                # Default to CSV parsing
                return self._parse_kev_csv(content.decode("utf-8"))

        except httpx.HTTPError as e:
            logger.warning("Failed to fetch KEV data from %s: %s", url, e)
            return {}
        except Exception as e:
            logger.warning("Failed to load KEV data from %s: %s", url, e)
            return {}

    def _parse_kev_csv(self, content: str) -> dict[str, dict[str, Any]]:
        """Parse KEV data from CSV content.

        Args:
            content: CSV content as string

        Returns:
            Dict mapping CVE ID to KEV entry dict
        """
        kev_data: dict[str, dict[str, Any]] = {}

        reader = csv.DictReader(io.StringIO(content))
        for row in reader:
            cve_id = row["cveID"]

            # Parse dates
            date_added = datetime.strptime(row["dateAdded"], "%Y-%m-%d").replace(
                tzinfo=UTC
            )
            due_date = datetime.strptime(row["dueDate"], "%Y-%m-%d").replace(tzinfo=UTC)

            kev_data[cve_id] = {
                "date_added": date_added,
                "due_date": due_date,
                "required_action": row.get("requiredAction"),
                "vendor": row.get("vendorProject"),
                "product": row.get("product"),
                "ransomware": row.get("knownRansomwareCampaignUse"),
                "vulnerability_name": row.get("vulnerabilityName"),
                "notes": row.get("notes"),
            }

        return kev_data

    def _parse_kev_json(self, content: bytes) -> dict[str, dict[str, Any]]:
        """Parse KEV data from JSON content.

        Args:
            content: JSON content as bytes

        Returns:
            Dict mapping CVE ID to KEV entry dict
        """
        kev_data: dict[str, dict[str, Any]] = {}

        data = json.loads(content)
        vulnerabilities = data.get("vulnerabilities", [])

        for vuln in vulnerabilities:
            cve_id = vuln.get("cveID")
            if not cve_id:
                continue

            # Parse dates
            date_added_str = vuln.get("dateAdded")
            due_date_str = vuln.get("dueDate")

            date_added = (
                datetime.strptime(date_added_str, "%Y-%m-%d").replace(tzinfo=UTC)
                if date_added_str
                else None
            )
            due_date = (
                datetime.strptime(due_date_str, "%Y-%m-%d").replace(tzinfo=UTC)
                if due_date_str
                else None
            )

            kev_data[cve_id] = {
                "date_added": date_added,
                "due_date": due_date,
                "required_action": vuln.get("requiredAction"),
                "vendor": vuln.get("vendorProject"),
                "product": vuln.get("product"),
                "ransomware": vuln.get("knownRansomwareCampaignUse"),
                "vulnerability_name": vuln.get("vulnerabilityName"),
                "notes": vuln.get("notes"),
            }

        return kev_data
