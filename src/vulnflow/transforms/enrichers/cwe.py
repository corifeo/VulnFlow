"""
CWE Enricher - MITRE CWE database enrichment

Provides:
- CWEEnricher: Enriches CWE entries with names, categories, and abstraction levels

Layer: Transform/Enrichers
Dependencies: dataset, utils.http, xml.etree
Used by: dataset.transform(), pipeline
"""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING, Any
from zipfile import ZipFile

import numpy as np

if TYPE_CHECKING:
    from vulnflow.dataset import VulnDataset

from vulnflow.registry import FieldDefinition, register_transform
from vulnflow.schema import CWEEntry
from vulnflow.transforms.base import BaseEnricher
from vulnflow.utils import json
from vulnflow.utils.http import HttpError, HttpFetcher, is_url

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
        cache_ttl: int = 0,
        cache_dir: Path | str = ".cache",
        timeout: float = 60.0,
    ) -> None:
        """Initialize enricher with CWE source.

        Args:
            source: Path to CWE XML file, URL, or None for default MITRE URL.
            cache_ttl: Cache time-to-live in seconds (0 = never expire)
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
        self._cache: dict[str, dict[str, Any]] | None = None
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
                        likelihood_of_exploit=metadata.get("likelihood_of_exploit"),
                        consequences=metadata.get("consequences") or None,
                        applicable_platforms=metadata.get("applicable_platforms")
                        or None,
                        modes_of_introduction=metadata.get("modes_of_introduction")
                        or None,
                    )
                    enriched_entries.append(cwe_entry.to_dict())
                    enriched_count += 1
                else:
                    # Keep original if not found in database
                    self.logger.debug("CWE %s not found in database", cwe_id)
                    enriched_entries.append(cwe_dict)

            enriched_cwes[i] = json.dumps(enriched_entries)

        # Update column
        dataset.set_column("cwes", enriched_cwes)
        self.logger.info("Enriched %d CWE entries across %d records", enriched_count, n)

    async def _get_cached_cwe_db(self) -> dict[str, dict[str, Any]]:
        """Get CWE database from cache or load fresh (async)."""
        now = time.time()
        expired = self.cache_ttl > 0 and now - self._cache_time > self.cache_ttl
        if self._cache is None or expired:
            self._cache = await self._load_cwe_database()
            self._cache_time = now
        return self._cache

    async def _load_cwe_database(self) -> dict[str, dict[str, Any]]:
        """Load CWE database from cache or source.

        Returns:
            Dict mapping CWE ID to metadata dict (fully denormalized)
        """
        mitre_dir = self.cache_dir / "mitre"
        mitre_dir.mkdir(parents=True, exist_ok=True)
        cache_file = mitre_dir / "cwe_database.json"
        capec_mapping_file = mitre_dir / "cwe_capec_mapping.json"

        # Use cached database if it exists and hasn't expired
        if cache_file.exists() and capec_mapping_file.exists():
            age = time.time() - cache_file.stat().st_mtime
            if self.cache_ttl == 0 or age < self.cache_ttl:
                self.logger.info("Using cached CWE database (age: %.0fs)", age)
                cached_db: dict[str, dict[str, Any]] = json.loads(
                    cache_file.read_text()
                )
                return cached_db

        # Load from source
        self.logger.info("Loading CWE database from %s", self.source)
        xml_path: Path | None
        if isinstance(self.source, Path):
            xml_path = self.source
        else:
            xml_path = await self._download_cwe_xml()
            if xml_path is None:
                return {}

        # Parse and cache
        cwe_db = self._parse_cwe_xml(xml_path)
        cache_file.write_text(json.dumps(cwe_db, indent=True))

        # Extract and cache CWE→CAPEC mapping
        capec_mapping = self._extract_capec_mapping(xml_path)
        capec_mapping_file.write_text(json.dumps(capec_mapping, indent=True))

        # Clean up intermediate files
        for name in ("cwec_latest.xml.zip", "cwec_latest.xml"):
            intermediate = mitre_dir / name
            if intermediate.exists():
                intermediate.unlink()

        self.logger.info("Cached %d CWE entries", len(cwe_db))
        return cwe_db

    async def _download_cwe_xml(self) -> Path | None:
        """Download CWE XML from URL.

        Returns:
            Path to extracted XML file
        """
        mitre_dir = self.cache_dir / "mitre"
        mitre_dir.mkdir(parents=True, exist_ok=True)

        if self._http_fetcher is None:
            self._http_fetcher = HttpFetcher(cache_dir=mitre_dir, timeout=self.timeout)

        # Download zip file
        zip_path = mitre_dir / "cwec_latest.xml.zip"
        try:
            content, _fmt = await self._http_fetcher.fetch(
                str(self.source),
                self.cache_ttl,
                progress_desc="Downloading CWE database",
            )
            zip_path.write_bytes(content)
        except HttpError as e:
            self.logger.warning(
                "Failed to download CWE XML from %s: %s", self.source, e
            )
            return None
        except Exception as e:
            self.logger.warning(
                "Failed to download CWE XML from %s: %s", self.source, e
            )
            return None

        # Extract XML
        xml_path = mitre_dir / "cwec_latest.xml"
        with ZipFile(zip_path, "r") as zf:
            # CWE zip contains single XML file
            xml_name = zf.namelist()[0]
            xml_path.write_bytes(zf.read(xml_name))

        return xml_path

    def _parse_cwe_xml(self, xml_path: Path) -> dict[str, dict[str, Any]]:
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
            self.logger.error("Failed to parse CWE XML file %s: %s", xml_path, e)
            raise ValueError(f"Invalid CWE XML format: {e}") from e

        # First pass: Build basic entries
        raw_db: dict[str, dict[str, Any]] = {}
        for weakness in root.findall(".//{*}Weakness"):
            cwe_id = f"CWE-{weakness.get('ID')}"
            category_id = self._extract_category_id(weakness)

            # Likelihood of exploit
            likelihood_el = weakness.find(".//{*}Likelihood_Of_Exploit")
            likelihood = likelihood_el.text if likelihood_el is not None else None

            # Consequences (scopes + impacts per consequence)
            consequences: list[dict[str, list[str]]] = []
            for cons_el in weakness.findall(".//{*}Consequence"):
                scopes = [s.text for s in cons_el.findall("{*}Scope") if s.text]
                impacts = [s.text for s in cons_el.findall("{*}Impact") if s.text]
                if scopes or impacts:
                    consequences.append({"scopes": scopes, "impacts": impacts})

            # Applicable platforms (Language, Operating_System, Architecture, Technology)
            platforms: list[str] = []
            ap_el = weakness.find(".//{*}Applicable_Platforms")
            if ap_el is not None:
                for child in ap_el:
                    name = child.get("Name")
                    if name:
                        platforms.append(name)

            # Modes of introduction (Phase text from Introduction children)
            modes: list[str] = []
            for intro_el in weakness.findall(
                ".//{*}Modes_Of_Introduction/{*}Introduction"
            ):
                phase_el = intro_el.find("{*}Phase")
                if phase_el is not None and phase_el.text:
                    modes.append(phase_el.text)

            raw_db[cwe_id] = {
                "name": weakness.get("Name", ""),
                "category_id": category_id,
                "category_name": None,  # Filled in second pass
                "abstraction_level": weakness.get("Abstraction", ""),
                "likelihood_of_exploit": likelihood,
                "consequences": consequences,
                "applicable_platforms": platforms,
                "modes_of_introduction": modes,
            }

        # Second pass: Add category names (requires all entries parsed)
        for _cwe_id, data in raw_db.items():
            category_id = data["category_id"]
            if category_id and isinstance(category_id, str) and category_id in raw_db:
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

    def _extract_capec_mapping(self, xml_path: Path) -> dict[str, list[str]]:
        """Extract CWE→CAPEC mapping from CWE XML.

        Returns:
            Dict mapping CWE ID to list of CAPEC IDs.
            Only includes CWEs that have at least one CAPEC mapping.
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()
        mapping: dict[str, list[str]] = {}
        for weakness in root.findall(".//{*}Weakness"):
            cwe_id = f"CWE-{weakness.get('ID')}"
            capec_ids = [
                f"CAPEC-{el.get('CAPEC_ID')}"
                for el in weakness.findall(".//{*}Related_Attack_Pattern")
                if el.get("CAPEC_ID")
            ]
            if capec_ids:
                mapping[cwe_id] = capec_ids
        return mapping
