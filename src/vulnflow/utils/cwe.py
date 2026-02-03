"""CWE Database - Load and query CWE database from XML.

Provides:
- CWEDatabase: In-memory CWE database for name and category lookups

Layer: Utility
Dependencies: None

Future Enhancement Potential:
    This module could be converted to a CWEEnricher using the official CWE database:
    - Source: Official CWE Database (https://cwe.mitre.org/data/downloads.html)
    - Format: XML with hierarchical structure (Category → Class/Base → Variant)
    - Latest Version: 4.19.1 (as of 2026)
    - Schema: https://cwe.mitre.org/data/xsd/cwe_schema_v6.0.xsd

    Potential enrichments:
    - Hierarchical grouping: Map CWE variants to parent categories
    - Weakness taxonomy: Classify weaknesses by abstraction level (Category/Class/Variant)
    - Related weaknesses: Identify sibling and cousin CWEs
    - Severity patterns: Analyze common exploit consequences by CWE family
    - Detection methods: Surface recommended detection strategies per CWE

    Implementation would require:
    - BaseEnricher subclass for async loading (XML is ~10MB compressed)
    - Local caching strategy with version checking
    - Graph traversal for hierarchical queries
    - Periodic updates (CWE releases quarterly)
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path


class CWEDatabase:
    """In-memory CWE database for name and category lookups.

    Attributes:
        _data: Internal storage mapping CWE ID to {name, category_id}

    Examples:
        >>> db = CWEDatabase.from_xml("cwec_latest.xml")
        >>> db.get_name("CWE-502")
        'Deserialization of Untrusted Data'
        >>> db.get_category("CWE-502")
        'CWE-913'
        >>> "CWE-79" in db
        True
        >>> len(db)
        926
    """

    def __init__(self) -> None:
        """Initialize empty CWE database.

        Use CWEDatabase.from_xml() to load from XML file.
        """
        self._data: dict[str, dict[str, str]] = {}

    def __len__(self) -> int:
        """Return number of CWEs in database.

        Returns:
            Count of CWE entries loaded
        """
        return len(self._data)

    def __contains__(self, cwe_id: str) -> bool:
        """Check if CWE exists in database.

        Args:
            cwe_id: CWE identifier (e.g., "CWE-79")

        Returns:
            True if CWE exists in database, False otherwise
        """
        return cwe_id in self._data

    @classmethod
    def from_xml(cls, path: str | Path) -> CWEDatabase:
        """Load CWE database from XML file.

        Parses official CWE XML format, extracting:
        - Weakness ID and Name
        - Parent category (first ChildOf relationship)

        Args:
            path: Path to CWE XML file (e.g., "cwec_latest.xml")

        Returns:
            CWEDatabase instance populated with weaknesses

        Raises:
            ET.ParseError: If XML file is malformed
            FileNotFoundError: If path does not exist

        Examples:
            >>> db = CWEDatabase.from_xml("data/cwec_v4.19.xml")
            >>> len(db)
            926
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
            cwe_id: CWE identifier (e.g., "CWE-502")

        Returns:
            Human-readable CWE name, or None if CWE not found

        Examples:
            >>> db.get_name("CWE-502")
            'Deserialization of Untrusted Data'
            >>> db.get_name("CWE-99999")
            None
        """
        entry = self._data.get(cwe_id)
        return entry["name"] if entry else None

    def get_category(self, cwe_id: str) -> str | None:
        """Get CWE category (parent) ID.

        Returns the immediate parent in the CWE hierarchy based on
        the first ChildOf relationship found in the XML.

        Args:
            cwe_id: CWE identifier (e.g., "CWE-502")

        Returns:
            Parent CWE ID, or None if CWE not found or has no parent

        Examples:
            >>> db.get_category("CWE-502")
            'CWE-913'
            >>> db.get_category("CWE-1000")  # Top-level category
            None
        """
        entry = self._data.get(cwe_id)
        return entry["category_id"] if entry else None
