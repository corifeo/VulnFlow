"""Schema introspection utilities.

Provides:
- get_record_columns: Get VulnRecord field names
- get_flattened_columns: Get flattened column names for DataFrame export

Layer: Utility
Dependencies: schema
Used by: extractors, loaders
"""

from __future__ import annotations

from dataclasses import fields


def get_record_columns() -> list[str]:
    """Get column names from VulnRecord dataclass.

    Extracts field names using dataclasses.fields() introspection.
    Used by extractors and loaders to ensure schema consistency.

    Returns:
        List of field names from VulnRecord schema in declaration order

    Examples:
        >>> cols = get_record_columns()
        >>> "cve_id" in cols
        True
        >>> "cvss_scores" in cols
        True
    """
    from vulnflow.schema import VulnRecord

    return [f.name for f in fields(VulnRecord)]


def get_flattened_columns() -> list[str]:
    """Get computed/flattened column names for DataFrame export.

    These columns are derived from VulnRecord nested fields during extraction,
    following flattening rules from DATA-STRUCTURES.md:
    - cvss_scores → cvss_base_score, cvss_version, cvss_vector_string, etc.
    - cwes → primary_cwe_id, primary_cwe_name
    - cpes → primary_vendor, primary_product, primary_version

    Returns:
        List of flattened column names in canonical order

    Examples:
        >>> cols = get_flattened_columns()
        >>> "cvss_base_score" in cols
        True
        >>> "primary_cwe_id" in cols
        True
        >>> len(cols)
        10
    """
    return [
        # CVSS flattened (from cvss_scores)
        "cvss_base_score",
        "cvss_base_severity",
        "cvss_version",
        "cvss_vector_string",
        "cvss_source",
        # CWE flattened (from cwes)
        "primary_cwe_id",
        "primary_cwe_name",
        # CPE flattened (from cpes)
        "primary_vendor",
        "primary_product",
        "primary_version",
    ]
