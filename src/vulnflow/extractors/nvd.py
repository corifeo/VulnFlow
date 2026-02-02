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
        with path.open(encoding="utf-8") as f:
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
            if not descriptions:
                continue

            for desc in descriptions:
                cwe_id = desc.get("value", "")  # type: ignore[union-attr]
                if cwe_id and cwe_id.startswith("CWE-"):  # type: ignore[union-attr]
                    cwes.append(
                        CWEEntry(
                            id=cwe_id,  # type: ignore[arg-type]
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
            if not nodes:
                continue

            for node in nodes:
                cpe_matches = node.get("cpeMatch", [])  # type: ignore[union-attr]
                if not cpe_matches:
                    continue

                for cpe_match in cpe_matches:
                    cpe_str = cpe_match.get("criteria", "")  # type: ignore[union-attr]
                    if cpe_str and cpe_str.startswith("cpe:2.3"):  # type: ignore[union-attr]
                        try:
                            cpe = parse_cpe23(cpe_str)  # type: ignore[arg-type]
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
            url = ref.get("url", "")  # type: ignore[union-attr]
            if url:
                source = ref.get("source")  # type: ignore[union-attr]
                tags = ref.get("tags", [])  # type: ignore[union-attr]
                ref_entries.append(
                    ReferenceEntry(
                        url=url,  # type: ignore[arg-type]
                        source=source if source else None,  # type: ignore[arg-type]
                        tags=tags if tags else [],  # type: ignore[arg-type]
                    )
                )

        return ref_entries
