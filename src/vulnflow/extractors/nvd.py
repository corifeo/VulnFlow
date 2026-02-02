"""
NVD Extractor - NVD JSON file reader

Provides:
- NVDExtractor: Extracts vulnerability records from NVD JSON files

Layer: Extract
Dependencies: schema
Used by: dataset.from_nvd()
"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from vulnflow.extractors.base import Extractor
from vulnflow.schema import VulnRecord


class NVDExtractor(Extractor):
    """Extracts vulnerability records from NVD JSON files."""

    name: str = "nvd"

    def __init__(self, path: str) -> None:
        self.path = path

    def extract(self) -> list[VulnRecord]:
        """Extract vulnerability records from NVD JSON file(s).

        Parses NVD 2.0 JSON format and creates minimal VulnRecords.
        Full parsing (CWE, CPE, CVSS, references) is implemented in Phase 9-10.
        """
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

            record = VulnRecord(
                id=uuid4(),
                cve_id=cve_id,
                description=description,
                sources=["nvd"],
            )
            records.append(record)

        return records
