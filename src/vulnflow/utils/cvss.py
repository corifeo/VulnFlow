"""
CVSS Utilities - Parsing and normalization for CVSS scores

Provides:
- normalize_source(): Email/domain → simple identifier
- extract_cvss_metadata(): Extract fields common to all CVSS versions

Layer: Utils
Dependencies: None
Used by: NVDExtractor, future vulnerability extractors
"""

from __future__ import annotations

# Source normalization map
CVSS_SOURCE_MAP = {
    "nvd": "nvd",
    "nist.gov": "nvd",
    "redhat.com": "redhat",
    "redhat": "redhat",
    "microsoft.com": "microsoft",
    "microsoft": "microsoft",
    "mitre.org": "mitre",
    "mitre": "mitre",
}


def normalize_source(source_raw: str) -> str:
    """Normalize CVSS source from email/domain to identifier.

    Examples:
        "nvd@nist.gov" → "nvd"
        "security@redhat.com" → "redhat"
        "unknown@example.com" → "example"

    Args:
        source_raw: Raw source string (usually email format)

    Returns:
        Normalized source identifier
    """
    source_lower = source_raw.lower()

    # Check known sources via dict lookup
    for key, value in CVSS_SOURCE_MAP.items():
        if key in source_lower:
            return value

    # Extract domain from email
    if "@" in source_raw:
        domain = source_raw.split("@")[1].split(".")[0]
        return domain.lower()

    return source_lower
