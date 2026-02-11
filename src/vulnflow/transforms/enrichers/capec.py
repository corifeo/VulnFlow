"""
CAPEC Enricher - MITRE CAPEC attack pattern enrichment

Provides:
- CAPECEnricher: Maps CVE→CWE→CAPEC to identify applicable attack patterns

Layer: Transform/Enrichers
Dependencies: dataset, utils.http, xml.etree, CWEEnricher
Used by: dataset.transform(), pipeline
"""

from __future__ import annotations

import logging
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from vulnflow.dataset import VulnDataset

from vulnflow.registry import FieldDefinition, register_transform
from vulnflow.schema import CAPECEntry
from vulnflow.transforms.base import BaseEnricher
from vulnflow.utils import json
from vulnflow.utils.http import HttpError, HttpFetcher, is_url

logger = logging.getLogger(__name__)

DEFAULT_CAPEC_URL = "https://capec.mitre.org/data/xml/capec_latest.xml"


@register_transform(default=False)
class CAPECEnricher(BaseEnricher):
    """Enrich vulnerabilities with CAPEC attack patterns via CWE mapping.

    BulkEnricher pattern: Downloads CAPEC XML, loads CWE→CAPEC mapping from
    CWEEnricher cache, merges bidirectional mappings, enriches records.
    """

    name: str = "CAPECEnricher"
    depends_on: list[str] = ["CWEEnricher"]

    OUTPUT_SCHEMA: list[FieldDefinition] = [
        FieldDefinition(
            "capecs", "str", "CAPEC attack patterns array (JSON)", "CAPECEnricher"
        ),
    ]

    def __init__(
        self,
        source: str | Path | None = None,
        cache_ttl: int = 0,
        cache_dir: Path | str = ".cache",
        timeout: float = 60.0,
    ) -> None:
        """Initialize enricher with CAPEC source.

        Args:
            source: Path to CAPEC XML file, URL, or None for default MITRE URL.
            cache_ttl: Cache time-to-live in seconds (0 = never expire)
            cache_dir: Directory for cache storage
            timeout: HTTP request timeout in seconds
        """
        super().__init__()

        if source is None:
            self.source: str | Path = DEFAULT_CAPEC_URL
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
        """Enrich records with CAPEC attack patterns based on CWE mappings.

        Args:
            dataset: VulnDataset to enrich (modifies in place)
        """
        capec_db = await self._get_cached_capec_db()

        # Build merged CWE→CAPEC mapping from both directions
        cwe_to_capec = self._build_cwe_to_capec_mapping(capec_db)

        # Get existing cwes column
        cwes_json = dataset.get_column("cwes")
        n = len(dataset)

        capecs_array = np.empty(n, dtype=object)
        enriched_count = 0

        for i, raw in enumerate(cwes_json):
            if not raw or raw == "[]" or raw == "null":
                capecs_array[i] = "[]"
                continue

            cwes_list = json.loads(str(raw))
            if not cwes_list:
                capecs_array[i] = "[]"
                continue

            # Collect all CAPEC IDs from all CWEs
            capec_ids: set[str] = set()
            for cwe_dict in cwes_list:
                cwe_id = cwe_dict.get("id")
                if cwe_id and cwe_id in cwe_to_capec:
                    capec_ids.update(cwe_to_capec[cwe_id])

            if not capec_ids:
                capecs_array[i] = "[]"
                continue

            # Build CAPECEntry for each
            entries = []
            for capec_id in sorted(capec_ids):
                metadata = capec_db.get(capec_id)
                if metadata:
                    entry = CAPECEntry(
                        id=capec_id,
                        name=metadata.get("name"),
                        abstraction=metadata.get("abstraction"),
                        status=metadata.get("status"),
                        likelihood_of_attack=metadata.get("likelihood_of_attack"),
                        typical_severity=metadata.get("typical_severity"),
                        skills_required=metadata.get("skills_required") or None,
                        consequences=metadata.get("consequences") or None,
                        attack_techniques=metadata.get("attack_techniques") or None,
                    )
                else:
                    entry = CAPECEntry(id=capec_id)
                entries.append(entry.to_dict())

            capecs_array[i] = json.dumps(entries)
            enriched_count += len(entries)

        dataset.set_column("capecs", capecs_array)
        logger.info(f"Enriched {enriched_count} CAPEC entries across {n} records")

    def _build_cwe_to_capec_mapping(
        self,
        capec_db: dict[str, dict[str, Any]],
    ) -> dict[str, set[str]]:
        """Build merged CWE→CAPEC mapping from both data sources.

        Combines:
        1. CWE XML mapping file (forward: CWE→CAPEC)
        2. CAPEC DB's related_cwes (reverse: CAPEC→CWE inverted)

        Returns:
            Dict mapping CWE ID to set of CAPEC IDs.
        """
        cwe_to_capec: dict[str, set[str]] = {}

        # Direction 1: CWE→CAPEC from CWE mapping file
        capec_mapping_file = self.cache_dir / "mitre" / "cwe_capec_mapping.json"
        if capec_mapping_file.exists():
            mapping: dict[str, list[str]] = json.loads(
                capec_mapping_file.read_text()
            )
            for cwe_id, capec_ids in mapping.items():
                cwe_to_capec.setdefault(cwe_id, set()).update(capec_ids)
        else:
            logger.warning(
                "CWE→CAPEC mapping file not found at %s, "
                "using only CAPEC→CWE direction",
                capec_mapping_file,
            )

        # Direction 2: CAPEC→CWE inverted to CWE→CAPEC
        for capec_id, metadata in capec_db.items():
            related_cwes = metadata.get("related_cwes")
            if isinstance(related_cwes, list):
                for cwe_id in related_cwes:
                    if isinstance(cwe_id, str):
                        cwe_to_capec.setdefault(cwe_id, set()).add(capec_id)

        return cwe_to_capec

    async def _get_cached_capec_db(
        self,
    ) -> dict[str, dict[str, Any]]:
        """Get CAPEC database from cache or load fresh (async)."""
        now = time.time()
        expired = self.cache_ttl > 0 and now - self._cache_time > self.cache_ttl
        if self._cache is None or expired:
            self._cache = await self._load_capec_database()
            self._cache_time = now
        return self._cache

    async def _load_capec_database(
        self,
    ) -> dict[str, dict[str, Any]]:
        """Load CAPEC database from cache or source.

        Returns:
            Dict mapping CAPEC ID to metadata dict.
        """
        mitre_dir = self.cache_dir / "mitre"
        mitre_dir.mkdir(parents=True, exist_ok=True)
        cache_file = mitre_dir / "capec_database.json"

        # Use cached database if it exists and hasn't expired
        if cache_file.exists():
            age = time.time() - cache_file.stat().st_mtime
            if self.cache_ttl == 0 or age < self.cache_ttl:
                logger.info(f"Using cached CAPEC database (age: {age:.0f}s)")
                cached_db: dict[str, dict[str, Any]] = json.loads(
                    cache_file.read_text()
                )
                return cached_db

        # Load from source
        logger.info(f"Loading CAPEC database from {self.source}")
        if isinstance(self.source, Path):
            xml_path = self.source
            capec_db = self._parse_capec_xml(xml_path)
        else:
            try:
                xml_path = await self._download_capec_xml()
                capec_db = self._parse_capec_xml(xml_path)
                # Clean up intermediate XML
                if xml_path.exists():
                    xml_path.unlink()
            except HttpError as e:
                logger.warning("Failed to download CAPEC XML: %s", e)
                return {}
            except Exception as e:
                logger.warning("Failed to download CAPEC XML: %s", e)
                return {}

        cache_file.write_text(json.dumps(capec_db, indent=True))
        logger.info(f"Cached {len(capec_db)} CAPEC entries")
        return capec_db

    async def _download_capec_xml(self) -> Path:
        """Download CAPEC XML from URL.

        Returns:
            Path to downloaded XML file.
        """
        mitre_dir = self.cache_dir / "mitre"
        mitre_dir.mkdir(parents=True, exist_ok=True)

        if self._http_fetcher is None:
            self._http_fetcher = HttpFetcher(
                cache_dir=mitre_dir, timeout=self.timeout
            )

        xml_path = mitre_dir / "capec_latest.xml"
        content, _fmt = await self._http_fetcher.fetch(
            str(self.source), self.cache_ttl, progress_desc="Downloading CAPEC database"
        )
        xml_path.write_bytes(content)
        return xml_path

    def _parse_capec_xml(
        self,
        xml_path: Path,
    ) -> dict[str, dict[str, Any]]:
        """Parse CAPEC XML and extract attack pattern metadata.

        Args:
            xml_path: Path to CAPEC XML file

        Returns:
            Dict mapping CAPEC ID to metadata with related CWEs.
        """
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except ET.ParseError as e:
            logger.error(f"Failed to parse CAPEC XML file {xml_path}: {e}")
            raise ValueError(f"Invalid CAPEC XML format: {e}") from e

        capec_db: dict[str, dict[str, Any]] = {}
        for attack_pattern in root.findall(".//{*}Attack_Pattern"):
            capec_id = f"CAPEC-{attack_pattern.get('ID')}"

            # Extract text from child elements
            likelihood_el = attack_pattern.find(".//{*}Likelihood_Of_Attack")
            severity_el = attack_pattern.find(".//{*}Typical_Severity")

            # Extract related CWEs
            related_cwes = [
                f"CWE-{el.get('CWE_ID')}"
                for el in attack_pattern.findall(".//{*}Related_Weakness")
                if el.get("CWE_ID")
            ]

            # Extract skills required (Level attribute from each Skill element)
            skills_required: list[str] = [
                el.get("Level", "")
                for el in attack_pattern.findall(".//{*}Skill")
                if el.get("Level")
            ]

            # Extract consequences (scopes + impacts per consequence)
            consequences: list[dict[str, list[str]]] = []
            for cons_el in attack_pattern.findall(".//{*}Consequence"):
                scopes = [
                    s.text
                    for s in cons_el.findall("{*}Scope")
                    if s.text
                ]
                impacts = [
                    s.text
                    for s in cons_el.findall("{*}Impact")
                    if s.text
                ]
                if scopes or impacts:
                    consequences.append({"scopes": scopes, "impacts": impacts})

            # Extract ATT&CK technique IDs (Entry_ID from ATTACK taxonomy)
            attack_techniques: list[str] = [
                entry_id_el.text
                for tm in attack_pattern.findall(".//{*}Taxonomy_Mapping")
                if tm.get("Taxonomy_Name") == "ATTACK"
                for entry_id_el in tm.findall("{*}Entry_ID")
                if entry_id_el.text
            ]

            capec_db[capec_id] = {
                "name": attack_pattern.get("Name"),
                "abstraction": attack_pattern.get("Abstraction"),
                "status": attack_pattern.get("Status"),
                "likelihood_of_attack": (
                    likelihood_el.text if likelihood_el is not None else None
                ),
                "typical_severity": (
                    severity_el.text if severity_el is not None else None
                ),
                "related_cwes": related_cwes,
                "skills_required": skills_required,
                "consequences": consequences,
                "attack_techniques": attack_techniques,
            }

        return capec_db
