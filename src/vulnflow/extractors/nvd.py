"""
NVD Extractor - NVD JSON file reader

Provides:
- NVDExtractor: Extracts vulnerability records from NVD JSON files

Layer: Extract
Dependencies: schema, utils
Used by: dataset.from_nvd(), pipeline
"""

from __future__ import annotations

import json
from dataclasses import is_dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import numpy as np

from vulnflow.extractors.base import BaseExtractor
from vulnflow.registry import Registry
from vulnflow.schema import CPEEntry, CVSSEntry, CWEEntry, ReferenceEntry, VulnRecord
from vulnflow.utils.cpe import parse_cpe23
from vulnflow.utils.schema import get_flattened_columns, get_record_columns

if TYPE_CHECKING:
    from vulnflow.dataset import VulnDataset


@Registry.register_extractor("nvd_json")
class NVDExtractor(BaseExtractor):
    """Extract from NVD JSON files → VulnDataset.

    Parses NVD 2.0 JSON format and converts directly to columnar storage.

    Conversion logic:
    - Parse JSON → intermediate VulnRecord structures
    - Build columnar storage (dict[str, np.ndarray])
    - Flatten CVSS/CWE/CPE to scalar columns
    - Apply NVD-specific CVSS source priority (nvd > redhat > microsoft)
    - Encode complex fields (lists) as JSON strings

    Does NOT populate CVD timestamps (v/f/d/p/x/a) - those come from EventInference.
    """

    name: str = "nvd"
    format: str = "nvd_json"

    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = path

    def extract(self) -> VulnDataset:
        """Extract from NVD JSON file → VulnDataset.

        Returns:
            VulnDataset with columnar storage populated

        Raises:
            FileNotFoundError: If NVD JSON file doesn't exist
        """

        path = Path(self.path)
        with path.open(encoding="utf-8") as f:
            data = json.load(f)

        # First pass: Parse to VulnRecord structures
        records: list[VulnRecord] = []
        vulnerabilities = data.get("vulnerabilities", [])

        for vuln in vulnerabilities:
            cve_data = vuln.get("cve", {})
            cve_id = cve_data.get("id")

            # Get English description
            description = None
            for desc in cve_data.get("descriptions", []):
                if desc.get("lang") == "en":
                    description = desc.get("value")
                    break

            # Parse NVD timestamps
            nvd_published_at = self._parse_timestamp(cve_data.get("published"))
            nvd_modified_at = self._parse_timestamp(cve_data.get("lastModified"))

            # Parse CVSS scores
            cvss_scores = self._parse_cvss(cve_data.get("metrics", {}))

            # Parse CWE weaknesses
            cwes = self._parse_cwes(cve_data.get("weaknesses", []))

            # Parse CPE configurations
            cpes = self._parse_cpes(cve_data.get("configurations", []))

            # Parse references
            references = self._parse_references(cve_data.get("references", []))

            record = VulnRecord(
                id=uuid4(),
                cve_id=cve_id,
                description=description,
                nvd_published_at=nvd_published_at,
                nvd_modified_at=nvd_modified_at,
                cvss_scores=cvss_scores,
                cwes=cwes,
                cpes=cpes,
                references=references,
                sources=["nvd"],
            )
            records.append(record)

        # Second pass: Convert to columnar storage
        return self._records_to_dataset(records)

    def _records_to_dataset(self, records: list[VulnRecord]) -> VulnDataset:
        """Convert list of VulnRecord to VulnDataset (columnar storage).

        This method owns the NVD-specific conversion logic including:
        - JSON encoding of complex fields
        - Flattening CVSS/CWE/CPE to scalar columns
        - NVD source priority for CVSS selection

        Args:
            records: List of parsed VulnRecord instances

        Returns:
            VulnDataset with columnar storage
        """
        from vulnflow.dataset import VulnDataset

        # Create JSON encoder once for reuse
        json_encoder = json.JSONEncoder(default=str)

        def dataclass_to_dict(obj: Any) -> Any:
            """Fast dataclass to dict conversion without asdict overhead."""
            if obj is None:
                return None
            if is_dataclass(obj) and not isinstance(obj, type):
                # Manually extract fields instead of using asdict()
                return {f.name: getattr(obj, f.name) for f in obj.__dataclass_fields__.values()}
            return obj

        def encode_nested(value: Any) -> str:
            """Encode nested structure as JSON string."""
            if value is None:
                return "null"
            if isinstance(value, list):
                # Fast path for empty lists
                if not value:
                    return "[]"
                # Convert dataclasses in list to dicts
                items = [dataclass_to_dict(item) for item in value]
                return json_encoder.encode(items)
            if is_dataclass(value) and not isinstance(value, type):
                return json_encoder.encode(dataclass_to_dict(value))
            return json_encoder.encode(value)

        def flatten_cvss(
            cvss_scores: list[CVSSEntry],
        ) -> dict[str, float | str | None]:
            """Flatten list[CVSSEntry] to scalar columns.

            Selection logic (NVD-specific):
            1. Prefer source: nvd > redhat > microsoft > other
            2. If tie, take first in list
            """
            if not cvss_scores:
                return {
                    "cvss_base_score": None,
                    "cvss_base_severity": None,
                    "cvss_version": None,
                    "cvss_vector_string": None,
                    "cvss_source": None,
                }

            # Source preference order (NVD domain knowledge)
            source_priority = {"nvd": 0, "redhat": 1, "microsoft": 2}

            def score_key(entry: CVSSEntry) -> int:
                return source_priority.get(entry.source, 999)

            # Select preferred entry
            selected = min(cvss_scores, key=score_key)

            return {
                "cvss_base_score": selected.base_score,
                "cvss_base_severity": selected.base_severity,
                "cvss_version": selected.version,
                "cvss_vector_string": selected.vector_string,
                "cvss_source": selected.source,
            }

        def flatten_cwe(cwes: list[CWEEntry]) -> dict[str, str | None]:
            """Flatten list[CWEEntry] to scalar columns (first entry)."""
            if not cwes:
                return {
                    "primary_cwe_id": None,
                    "primary_cwe_name": None,
                }
            # Take first CWE
            first = cwes[0]
            return {
                "primary_cwe_id": first.id,
                "primary_cwe_name": first.name,
            }

        def flatten_cpe(cpes: list[CPEEntry]) -> dict[str, str | None]:
            """Flatten list[CPEEntry] to scalar columns (first entry)."""
            if not cpes:
                return {
                    "primary_vendor": None,
                    "primary_product": None,
                    "primary_version": None,
                }
            # Take first CPE
            first = cpes[0]
            return {
                "primary_vendor": first.vendor,
                "primary_product": first.product,
                "primary_version": first.version,
            }

        dataset = VulnDataset()
        if not records:
            return dataset

        # Pre-allocate columnar arrays
        n = len(records)
        # Get all column names from VulnRecord + flattened columns
        record_cols = get_record_columns()
        flattened_cols = get_flattened_columns()
        all_columns = record_cols + flattened_cols
        arrays = {col: np.empty(n, dtype=object) for col in all_columns}

        # Single pass over records
        for i, r in enumerate(records):
            # Identity
            arrays["id"][i] = str(r.id)
            arrays["cve_id"][i] = r.cve_id
            arrays["aliases"][i] = json.dumps(r.aliases)
            # Description
            arrays["title"][i] = r.title
            arrays["description"][i] = r.description
            # Metadata
            arrays["first_seen_at"][i] = r.first_seen_at
            arrays["last_updated_at"][i] = r.last_updated_at
            arrays["sources"][i] = json.dumps(r.sources)
            # NVD Temporal
            arrays["nvd_published_at"][i] = r.nvd_published_at
            arrays["nvd_modified_at"][i] = r.nvd_modified_at
            # Scoring
            arrays["cvss_scores"][i] = encode_nested(r.cvss_scores)
            arrays["epss_scores"][i] = encode_nested(r.epss_scores)
            # Flattened CVSS
            flattened = flatten_cvss(r.cvss_scores)
            arrays["cvss_base_score"][i] = flattened["cvss_base_score"]
            arrays["cvss_base_severity"][i] = flattened["cvss_base_severity"]
            arrays["cvss_version"][i] = flattened["cvss_version"]
            arrays["cvss_vector_string"][i] = flattened["cvss_vector_string"]
            arrays["cvss_source"][i] = flattened["cvss_source"]
            # Exploitation
            arrays["kev"][i] = encode_nested(r.kev)
            arrays["exploits"][i] = encode_nested(r.exploits)
            # Classification
            arrays["cwes"][i] = encode_nested(r.cwes)
            arrays["cpes"][i] = encode_nested(r.cpes)
            # Flattened CWE/CPE
            cwe_flat = flatten_cwe(r.cwes)
            arrays["primary_cwe_id"][i] = cwe_flat["primary_cwe_id"]
            arrays["primary_cwe_name"][i] = cwe_flat["primary_cwe_name"]
            cpe_flat = flatten_cpe(r.cpes)
            arrays["primary_vendor"][i] = cpe_flat["primary_vendor"]
            arrays["primary_product"][i] = cpe_flat["primary_product"]
            arrays["primary_version"][i] = cpe_flat["primary_version"]
            # References
            arrays["references"][i] = encode_nested(r.references)
            # Extensible
            arrays["tags"][i] = json.dumps(r.tags)
            arrays["metadata"][i] = json.dumps(r.metadata, default=str)

        dataset._data = arrays

        return dataset

    def _parse_timestamp(self, ts: str | None) -> datetime | None:
        """Parse ISO timestamp string to datetime."""
        if not ts:
            return None
        # Handle Z suffix
        ts = ts.replace("Z", "+00:00")
        return datetime.fromisoformat(ts)

    def _parse_cvss(self, metrics: dict[str, list[dict]]) -> list[CVSSEntry]:  # type: ignore[type-arg]
        """Parse CVSS metrics from NVD format with multi-source support."""
        cvss_scores: list[CVSSEntry] = []

        # CVSS 3.1
        for metric in metrics.get("cvssMetricV31", []):
            cvss_data = metric.get("cvssData", {})
            # Parse source from metric (e.g., "nvd@nist.gov", "security@vendor.com")
            source_raw = metric.get("source", "nvd@nist.gov")
            # Normalize source to simple identifier (nvd, redhat, microsoft, etc.)
            source = self._normalize_cvss_source(source_raw)

            cvss_scores.append(
                CVSSEntry(
                    version=cvss_data.get("version", "3.1"),
                    vector_string=cvss_data.get("vectorString", ""),
                    base_score=cvss_data.get("baseScore", 0.0),
                    base_severity=cvss_data.get("baseSeverity", "NONE"),
                    source=source,
                )
            )

        # CVSS 3.0 (if present)
        for metric in metrics.get("cvssMetricV30", []):
            cvss_data = metric.get("cvssData", {})
            source_raw = metric.get("source", "nvd@nist.gov")
            source = self._normalize_cvss_source(source_raw)

            cvss_scores.append(
                CVSSEntry(
                    version=cvss_data.get("version", "3.0"),
                    vector_string=cvss_data.get("vectorString", ""),
                    base_score=cvss_data.get("baseScore", 0.0),
                    base_severity=cvss_data.get("baseSeverity", "NONE"),
                    source=source,
                )
            )

        # CVSS 2.0 (if present)
        for metric in metrics.get("cvssMetricV2", []):
            cvss_data = metric.get("cvssData", {})
            source_raw = metric.get("source", "nvd@nist.gov")
            source = self._normalize_cvss_source(source_raw)

            cvss_scores.append(
                CVSSEntry(
                    version=cvss_data.get("version", "2.0"),
                    vector_string=cvss_data.get("vectorString", ""),
                    base_score=cvss_data.get("baseScore", 0.0),
                    base_severity=metric.get("baseSeverity", "NONE"),
                    source=source,
                )
            )

        return cvss_scores

    def _normalize_cvss_source(self, source_raw: str) -> str:
        """Normalize CVSS source from email format to simple identifier.

        Examples:
            "nvd@nist.gov" -> "nvd"
            "security@redhat.com" -> "redhat"
            "security@microsoft.com" -> "microsoft"
            "cve@mitre.org" -> "mitre"

        Args:
            source_raw: Raw source string from NVD (usually email format)

        Returns:
            Normalized source identifier
        """
        # Default mapping for known sources
        if "nvd" in source_raw.lower() or "nist.gov" in source_raw.lower():
            return "nvd"
        elif "redhat" in source_raw.lower():
            return "redhat"
        elif "microsoft" in source_raw.lower():
            return "microsoft"
        elif "mitre" in source_raw.lower():
            return "mitre"
        else:
            # Extract domain or username before @
            # "security@example.com" -> "example"
            if "@" in source_raw:
                parts = source_raw.split("@")
                if len(parts) > 1:
                    domain = parts[1].split(".")[0]
                    return domain.lower()
            return source_raw.lower()

    def _parse_cwes(
        self, weaknesses: list[dict[str, object]]
    ) -> list[CWEEntry]:
        """Parse CWE weaknesses from NVD format."""
        cwes: list[CWEEntry] = []

        for weakness in weaknesses:
            descriptions = weakness.get("description", [])
            if not descriptions or not isinstance(descriptions, list):
                continue

            for desc in descriptions:
                if not isinstance(desc, dict):
                    continue
                cwe_id = desc.get("value", "")
                if cwe_id and isinstance(cwe_id, str) and cwe_id.startswith("CWE-"):
                    cwes.append(
                        CWEEntry(
                            id=cwe_id,
                            name=None,
                            category_id=None,
                        )
                    )

        return cwes

    def _parse_cpes(
        self, configurations: list[dict[str, object]]
    ) -> list[CPEEntry]:
        """Parse CPE configurations from NVD format."""
        cpes: list[CPEEntry] = []

        for config in configurations:
            nodes = config.get("nodes", [])
            if not nodes or not isinstance(nodes, list):
                continue

            for node in nodes:
                if not isinstance(node, dict):
                    continue
                cpe_matches = node.get("cpeMatch", [])
                if not cpe_matches or not isinstance(cpe_matches, list):
                    continue

                for cpe_match in cpe_matches:
                    if not isinstance(cpe_match, dict):
                        continue
                    cpe_str = cpe_match.get("criteria", "")
                    if cpe_str and isinstance(cpe_str, str) and cpe_str.startswith("cpe:2.3"):
                        try:
                            cpe = parse_cpe23(cpe_str)
                            cpes.append(cpe)
                        except (ValueError, AttributeError):
                            # Skip invalid CPE strings or malformed data
                            pass

        return cpes

    def _parse_references(
        self, references: list[dict[str, object]]
    ) -> list[ReferenceEntry]:
        """Parse references from NVD format."""
        ref_entries: list[ReferenceEntry] = []

        for ref in references:
            url = ref.get("url", "")
            if url and isinstance(url, str):
                source = ref.get("source")
                tags = ref.get("tags", [])
                ref_entries.append(
                    ReferenceEntry(
                        url=url,
                        source=source if isinstance(source, str) else None,
                        tags=tags if isinstance(tags, list) else [],
                    )
                )

        return ref_entries
