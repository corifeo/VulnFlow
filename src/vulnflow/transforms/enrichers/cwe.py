"""
CWE Enricher - MITRE CWE database enrichment

Provides:
- CWEEnricher: Enriches CWE entries with names, categories, and abstraction levels

Layer: Transform/Enrichers
Dependencies: dataset, utils.http, xml.etree
Used by: dataset.transform(), pipeline
"""

from __future__ import annotations

import json
import logging
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from zipfile import ZipFile

import numpy as np

if TYPE_CHECKING:
    from vulnflow.dataset import VulnDataset

from vulnflow.registry import FieldDefinition, register_transform
from vulnflow.schema import CWEEntry
from vulnflow.transforms.base import BaseEnricher
from vulnflow.utils.http import HttpFetcher, is_url

logger = logging.getLogger(__name__)

DEFAULT_CWE_URL = "https://cwe.mitre.org/data/xml/cwec_latest.xml.zip"


@register_transform(default=True)
class CWEEnricher(BaseEnricher):
    """Enrich CWE entries with metadata from MITRE CWE database.

    BulkEnricher pattern: Downloads CWE XML, caches parsed database,
    enriches existing cwes array with names, categories, and abstraction levels.
    """

    name: str = "CWEEnricher"
    depends_on: list[str] = []

    OUTPUT_SCHEMA: list[FieldDefinition] = [
        FieldDefinition("cwes", "str", "CWE entries array (JSON)", "CWEEnricher"),
    ]

    def __init__(
        self,
        source: str | Path | None = None,
        cache_ttl: int = 2592000,  # 30 days (CWE updates quarterly)
        cache_dir: Path | str = ".cache",
        timeout: float = 60.0,
    ) -> None:
        """Initialize enricher with CWE source.

        Args:
            source: Path to CWE XML file, URL, or None for default MITRE URL.
            cache_ttl: Cache time-to-live in seconds (default 30 days)
            cache_dir: Directory for cache storage
            timeout: HTTP request timeout in seconds
        """
        super().__init__()

        # Determine source type
        if source is None:
            self.source: str | Path = DEFAULT_CWE_URL
        elif isinstance(source, str) and is_url(source):
            self.source = source
        else:
            self.source = Path(source)

        self.cache_ttl = cache_ttl
        self.cache_dir = Path(cache_dir)
        self.timeout = timeout
        self._cache: dict[str, dict[str, str | None]] | None = None
        self._cache_time: float = 0.0
        self._http_fetcher: HttpFetcher | None = None

    async def transform(self, dataset: VulnDataset) -> None:  # type: ignore[override]
        """Enrich existing cwes array with CWE metadata.

        Args:
            dataset: VulnDataset to enrich (modifies in place)
        """
        # Load CWE database with caching
        cwe_db = await self._get_cached_cwe_db()

        # Get existing cwes column
        cwes_json = dataset.get_column("cwes")
        n = len(dataset)

        # Enrich each record's CWE entries
        enriched_cwes = np.empty(n, dtype=object)
        enriched_count = 0

        for i, raw in enumerate(cwes_json):
            if not raw or raw == "[]" or raw == "null":
                enriched_cwes[i] = "[]"
                continue

            cwes_list = json.loads(str(raw))
            if not cwes_list:
                enriched_cwes[i] = "[]"
                continue

            enriched_entries = []

            for cwe_dict in cwes_list:
                cwe_id = cwe_dict.get("id")
                if cwe_id and cwe_id in cwe_db:
                    metadata = cwe_db[cwe_id]
                    # Create fully populated CWEEntry
                    cwe_entry = CWEEntry(
                        id=cwe_id,
                        name=metadata.get("name"),
                        category_id=metadata.get("category_id"),
                        category_name=metadata.get("category_name"),
                        abstraction_level=metadata.get("abstraction_level"),
                    )
                    enriched_entries.append(asdict(cwe_entry))
                    enriched_count += 1
                else:
                    # Keep original if not found in database
                    logger.debug(f"CWE {cwe_id} not found in database")
                    enriched_entries.append(cwe_dict)

            enriched_cwes[i] = json.dumps(enriched_entries)

        # Update column
        dataset.set_column("cwes", enriched_cwes)
        logger.info(f"Enriched {enriched_count} CWE entries across {n} records")

    async def _get_cached_cwe_db(self) -> dict[str, dict[str, str | None]]:
        """Get CWE database from cache or load fresh (async)."""
        now = time.time()
        if self._cache is None or now - self._cache_time > self.cache_ttl:
            self._cache = await self._load_cwe_database()
            self._cache_time = now
        return self._cache

    async def _load_cwe_database(self) -> dict[str, dict[str, str | None]]:
        """Load CWE database from cache or source.

        Returns:
            Dict mapping CWE ID to metadata dict (fully denormalized)
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self.cache_dir / "cwe_database.json"
        cache_meta = self.cache_dir / "cwe_database.meta"

        # Check cache freshness
        if cache_file.exists() and cache_meta.exists():
            meta = json.loads(cache_meta.read_text())
            cached_time = datetime.fromisoformat(meta["downloaded_at"])
            age_seconds = (datetime.now() - cached_time).total_seconds()

            if age_seconds < self.cache_ttl:
                logger.info(f"Using cached CWE database (age: {age_seconds:.0f}s)")
                cached_db: dict[str, dict[str, str | None]] = json.loads(
                    cache_file.read_text()
                )
                return cached_db

        # Load from source
        logger.info(f"Loading CWE database from {self.source}")
        if isinstance(self.source, Path):
            xml_path = self.source
        else:
            xml_path = await self._download_cwe_xml()

        # Parse and cache
        cwe_db = self._parse_cwe_xml(xml_path)
        cache_file.write_text(json.dumps(cwe_db, indent=2))
        cache_meta.write_text(
            json.dumps({"downloaded_at": datetime.now().isoformat()})
        )

        logger.info(f"Cached {len(cwe_db)} CWE entries")
        return cwe_db

    async def _download_cwe_xml(self) -> Path:
        """Download CWE XML from URL.

        Returns:
            Path to extracted XML file
        """
        if self._http_fetcher is None:
            self._http_fetcher = HttpFetcher(
                cache_dir=self.cache_dir, timeout=self.timeout
            )

        # Download zip file
        zip_path = self.cache_dir / "cwec_latest.xml.zip"
        try:
            content, _fmt = await self._http_fetcher.fetch(str(self.source), cache_ttl=self.cache_ttl)
            zip_path.write_bytes(content)
        except Exception as e:
            logger.error(f"Failed to download CWE XML from {self.source}: {e}")
            raise RuntimeError(f"Failed to download CWE database: {e}") from e

        # Extract XML
        xml_path = self.cache_dir / "cwec_latest.xml"
        with ZipFile(zip_path, "r") as zf:
            # CWE zip contains single XML file
            xml_name = zf.namelist()[0]
            xml_path.write_bytes(zf.read(xml_name))

        return xml_path

    def _parse_cwe_xml(self, xml_path: Path) -> dict[str, dict[str, str | None]]:
        """Parse CWE XML and pre-compute all fields (slow, done once).

        Args:
            xml_path: Path to CWE XML file

        Returns:
            Fully denormalized dict mapping CWE ID to metadata
        """
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except ET.ParseError as e:
            logger.error(f"Failed to parse CWE XML file {xml_path}: {e}")
            raise ValueError(f"Invalid CWE XML format: {e}") from e

        # First pass: Build basic entries
        raw_db: dict[str, dict[str, str | None]] = {}
        for weakness in root.findall(".//{*}Weakness"):
            cwe_id = f"CWE-{weakness.get('ID')}"
            category_id = self._extract_category_id(weakness)

            raw_db[cwe_id] = {
                "name": weakness.get("Name", ""),
                "category_id": category_id,
                "category_name": None,  # Filled in second pass
                "abstraction_level": weakness.get("Abstraction", ""),
            }

        # Second pass: Add category names (requires all entries parsed)
        for _cwe_id, data in raw_db.items():
            category_id = data["category_id"]
            if category_id and category_id in raw_db:
                data["category_name"] = raw_db[category_id]["name"]

        return raw_db

    def _extract_category_id(self, weakness: ET.Element) -> str | None:
        """Extract parent category ID from weakness element.

        Args:
            weakness: XML element for weakness

        Returns:
            Parent CWE ID (first ChildOf relationship) or None
        """
        for related in weakness.findall(".//{*}Related_Weakness[@Nature='ChildOf']"):
            return f"CWE-{related.get('CWE_ID')}"
        return None
