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

from typing import Any

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


def extract_cvss_metadata(metric: dict[str, Any]) -> dict[str, Any]:
    """Extract fields common to all CVSS versions from NVD metric.

    Handles:
    - Source normalization
    - Authority flag (Primary vs Secondary)
    - Exploitability and impact scores

    Args:
        metric: NVD CVSS metric dict

    Returns:
        Dict with common fields ready for CVSSEntry(**metadata) unpacking
    """
    source_raw = metric.get("source", "nvd@nist.gov")

    return {
        "source": normalize_source(source_raw),
        "is_authoritative": metric.get("type") == "Primary",
        "exploitability_score": metric.get("exploitabilityScore"),
        "impact_score": metric.get("impactScore"),
        "source_url": None,  # Not in NVD data
        "published_at": None,  # Not in NVD data
        "updated_at": None,  # Not in NVD data
    }
