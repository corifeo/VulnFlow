"""
CWE Database - Load and query CWE database from XML

Provides:
- CWEDatabase: In-memory CWE database for name and category lookups

Layer: Utility
Dependencies: None
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path


class CWEDatabase:
    """In-memory CWE database for name and category lookups.

    Usage:
        db = CWEDatabase.from_xml("cwec_latest.xml")
        name = db.get_name("CWE-502")  # "Deserialization of Untrusted Data"
        category = db.get_category("CWE-502")  # "CWE-913"
    """

    def __init__(self) -> None:
        """Initialize empty CWE database."""
        self._data: dict[str, dict[str, str]] = {}

    def __len__(self) -> int:
        """Return number of CWEs in database."""
        return len(self._data)

    def __contains__(self, cwe_id: str) -> bool:
        """Check if CWE exists in database."""
        return cwe_id in self._data

    @classmethod
    def from_xml(cls, path: str | Path) -> CWEDatabase:
        """Load CWE database from XML file.

        Args:
            path: Path to CWE XML file (cwec_latest.xml)

        Returns:
            CWEDatabase instance
        """
        db = cls()
        tree = ET.parse(path)
        root = tree.getroot()

        for weakness in root.findall(".//Weakness"):
            cwe_id = f"CWE-{weakness.get('ID')}"
            name = weakness.get("Name", "")

            # Extract parent category (first ChildOf relationship)
            category_id: str | None = None
            for related in weakness.findall(".//Related_Weakness[@Nature='ChildOf']"):
                category_id = f"CWE-{related.get('CWE_ID')}"
                break

            db._data[cwe_id] = {"name": name, "category_id": category_id or ""}

        return db

    def get_name(self, cwe_id: str) -> str | None:
        """Get CWE name by ID.

        Args:
            cwe_id: CWE ID (e.g., "CWE-502")

        Returns:
            CWE name or None if not found
        """
        entry = self._data.get(cwe_id)
        return entry["name"] if entry else None

    def get_category(self, cwe_id: str) -> str | None:
        """Get CWE category (parent) ID.

        Args:
            cwe_id: CWE ID (e.g., "CWE-502")

        Returns:
            Parent CWE ID or None if not found
        """
        entry = self._data.get(cwe_id)
        return entry["category_id"] if entry else None
