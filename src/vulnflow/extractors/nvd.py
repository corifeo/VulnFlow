"""
NVD Extractor - NVD JSON file reader

Provides:
- NVDExtractor: Extracts vulnerability records from NVD JSON files

Layer: Extract
Dependencies: schema
Used by: dataset.from_nvd(), pipeline
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from vulnflow.extractors.base import Extractor
from vulnflow.schema import CPEEntry, CVSSEntry, CWEEntry, ReferenceEntry, VulnRecord
from vulnflow.utils.cpe import parse_cpe23


class NVDExtractor(Extractor):
    """Extracts vulnerability records from NVD JSON files.

    Parses NVD 2.0 JSON format. Populates:
    - cve_id, description
    - nvd_published_at, nvd_modified_at (NVD temporal)
    - cvss_scores (CVSS 3.1 entries)

    Does NOT populate CVD timestamps (v/f/d/p/x/a) - those come from EventInference.
    """

    name: str = "nvd"

    def __init__(self, path: str) -> None:
        self.path = path

    def extract(self) -> list[VulnRecord]:
        """Extract vulnerability records from NVD JSON file(s)."""
        path = Path(self.path)
        with path.open() as f:
            data = json.load(f)

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

        return records

    def _parse_timestamp(self, ts: str | None) -> datetime | None:
        """Parse ISO timestamp string to datetime."""
        if not ts:
            return None
        # Handle Z suffix
        ts = ts.replace("Z", "+00:00")
        return datetime.fromisoformat(ts)

    def _parse_cvss(self, metrics: dict[str, list[dict]]) -> list[CVSSEntry]:  # type: ignore[type-arg]
        """Parse CVSS metrics from NVD format."""
        cvss_scores: list[CVSSEntry] = []

        # CVSS 3.1
        for metric in metrics.get("cvssMetricV31", []):
            cvss_data = metric.get("cvssData", {})
            cvss_scores.append(
                CVSSEntry(
                    version=cvss_data.get("version", "3.1"),
                    vector_string=cvss_data.get("vectorString", ""),
                    base_score=cvss_data.get("baseScore", 0.0),
                    base_severity=cvss_data.get("baseSeverity", "NONE"),
                    source="nvd",
                )
            )

        # CVSS 3.0 (if present)
        for metric in metrics.get("cvssMetricV30", []):
            cvss_data = metric.get("cvssData", {})
            cvss_scores.append(
                CVSSEntry(
                    version=cvss_data.get("version", "3.0"),
                    vector_string=cvss_data.get("vectorString", ""),
                    base_score=cvss_data.get("baseScore", 0.0),
                    base_severity=cvss_data.get("baseSeverity", "NONE"),
                    source="nvd",
                )
            )

        # CVSS 2.0 (if present)
        for metric in metrics.get("cvssMetricV2", []):
            cvss_data = metric.get("cvssData", {})
            cvss_scores.append(
                CVSSEntry(
                    version=cvss_data.get("version", "2.0"),
                    vector_string=cvss_data.get("vectorString", ""),
                    base_score=cvss_data.get("baseScore", 0.0),
                    base_severity=metric.get("baseSeverity", "NONE"),
                    source="nvd",
                )
            )

        return cvss_scores

    def _parse_cwes(
        self, weaknesses: list[dict[str, object]]
    ) -> list[CWEEntry]:
        """Parse CWE weaknesses from NVD format."""
        cwes: list[CWEEntry] = []

        for weakness in weaknesses:
            descriptions = weakness.get("description", [])
            if not isinstance(descriptions, list):
                continue

            for desc in descriptions:
                if not isinstance(desc, dict):
                    continue
                cwe_id = desc.get("value", "")
                if isinstance(cwe_id, str) and cwe_id.startswith("CWE-"):
                    cwes.append(
                        CWEEntry(
                            id=cwe_id,
                            name=None,  # Would need CWE database for name lookup
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
            if not isinstance(nodes, list):
                continue

            for node in nodes:
                if not isinstance(node, dict):
                    continue
                cpe_matches = node.get("cpeMatch", [])
                if not isinstance(cpe_matches, list):
                    continue

                for cpe_match in cpe_matches:
                    if not isinstance(cpe_match, dict):
                        continue
                    cpe_str = cpe_match.get("criteria", "")
                    if isinstance(cpe_str, str) and cpe_str.startswith("cpe:2.3"):
                        try:
                            cpe = parse_cpe23(cpe_str)
                            cpes.append(cpe)
                        except ValueError:
                            # Skip invalid CPE strings
                            pass

        return cpes

    def _parse_references(
        self, references: list[dict[str, object]]
    ) -> list[ReferenceEntry]:
        """Parse references from NVD format."""
        ref_entries: list[ReferenceEntry] = []

        for ref in references:
            if not isinstance(ref, dict):
                continue

            url = ref.get("url", "")
            source = ref.get("source")
            tags = ref.get("tags", [])

            if isinstance(url, str) and url:
                ref_entries.append(
                    ReferenceEntry(
                        url=url,
                        source=source if isinstance(source, str) else None,
                        tags=tags if isinstance(tags, list) else [],
                    )
                )

        return ref_entries
