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

import numpy as np

if TYPE_CHECKING:
    from vulnflow.dataset import VulnDataset

from vulnflow.registry import FieldDefinition, register_transform
from vulnflow.transforms.base import BaseEnricher
from vulnflow.utils import json
from vulnflow.utils.http import HttpError, HttpFetcher, is_url

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
        FieldDefinition("kev", "str", "KEV entry (JSON object)", "KEVEnricher"),
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
        """Enrich dataset with KEV data.

        Args:
            dataset: VulnDataset to enrich (modifies in place)
        """
        from vulnflow.schema import KEVEntry

        # Load KEV catalog with caching
        kev_data = await self._get_cached_data_async()

        # Get CVE IDs from dataset
        cve_ids = dataset.get_column("cve_id")
        n = len(dataset)

        # Initialize output
        kev_objects = np.full(n, None, dtype=object)

        # Populate KEV entries
        for i, cve_id in enumerate(cve_ids):
            if cve_id in kev_data:
                data = kev_data[cve_id]
                kev_entry = KEVEntry(
                    date_added=data["date_added"],
                    due_date=data["due_date"],
                    vendor=data["vendor"],
                    product=data["product"],
                    vulnerability_name=data.get("vulnerability_name", ""),
                    required_action=data["required_action"],
                    known_ransomware_use=data.get("ransomware") == "Known",
                )
                # Convert to dict with ISO datetime format
                kev_dict = {
                    "date_added": kev_entry.date_added.isoformat(),
                    "due_date": kev_entry.due_date.isoformat()
                    if kev_entry.due_date
                    else None,
                    "vendor": kev_entry.vendor,
                    "product": kev_entry.product,
                    "vulnerability_name": kev_entry.vulnerability_name,
                    "required_action": kev_entry.required_action,
                    "known_ransomware_use": kev_entry.known_ransomware_use,
                }
                kev_objects[i] = json.dumps(kev_dict)
            else:
                kev_objects[i] = "null"

        # Set JSON column
        dataset.set_column("kev", kev_objects)

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
            content, fmt = await self._http_fetcher.fetch(
                url, self.cache_ttl, progress_desc="Downloading KEV catalog"
            )

            if fmt == "json":
                return self._parse_kev_json(content)
            else:
                # Default to CSV parsing
                return self._parse_kev_csv(content.decode("utf-8"))

        except HttpError as e:
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
            cve_id = row.get("cveID")
            if not cve_id:
                continue

            # Parse dates
            date_added_str = row.get("dateAdded")
            due_date_str = row.get("dueDate")
            if not date_added_str:
                continue
            date_added = datetime.strptime(date_added_str, "%Y-%m-%d").replace(
                tzinfo=UTC
            )
            due_date = (
                datetime.strptime(due_date_str, "%Y-%m-%d").replace(tzinfo=UTC)
                if due_date_str
                else None
            )

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
