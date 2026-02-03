"""CPE Parser - Parse CPE 2.3 formatted strings.

Provides:
- parse_cpe23: Parse CPE 2.3 string into CPEEntry

Layer: Utility
Dependencies: schema

Future Enhancement Potential:
    This module could be converted to a CPEEnricher using the NVD CPE Dictionary:
    - Source: Official NVD CPE Dictionary (https://nvd.nist.gov/products/cpe)
    - Size: ~63MB, contains 1.5M+ CPE names and 420K+ match strings
    - API: https://nvd.nist.gov/developers/products (preferred, updated every 2 hours)
    - Data Feeds: JSON format, updated daily

    Potential enrichments:
    - Product taxonomy: Classify vendors/products into categories
    - Version analysis: Identify EOL/unsupported versions
    - Affected product grouping: Link related products/vendors
    - CPE match validation: Verify CPE strings against official dictionary

    Implementation would require:
    - BaseEnricher subclass with async I/O for API calls
    - Local caching strategy (63MB dictionary)
    - Rate limiting for API compliance
    - Fallback to data feeds if API unavailable
"""

from __future__ import annotations

from vulnflow.schema import CPEEntry


def parse_cpe23(cpe_str: str) -> CPEEntry:
    """Parse CPE 2.3 formatted string into structured CPEEntry.

    CPE 2.3 format:
        cpe:2.3:part:vendor:product:version:update:edition:language:sw_edition:target_sw:target_hw:other

    Args:
        cpe_str: CPE 2.3 formatted string (e.g., "cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*:*")

    Returns:
        CPEEntry with parsed vendor, product, version fields

    Raises:
        ValueError: If cpe_str is empty or not valid CPE 2.3 format

    Examples:
        >>> parse_cpe23("cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*:*")
        CPEEntry(vendor='apache', product='log4j', version='2.14.1', raw='...')
    """
    if not cpe_str:
        raise ValueError("CPE string cannot be empty")

    parts = cpe_str.split(":")

    # Valid CPE 2.3: cpe:2.3:part:vendor:product:version:...
    # Minimum 6 parts required (up to version field)
    if len(parts) < 6:
        raise ValueError(
            f"Invalid CPE 2.3 format: expected at least 6 colon-separated parts, got {len(parts)} in '{cpe_str}'"
        )

    if parts[0] != "cpe":
        raise ValueError(
            f"Invalid CPE 2.3 format: expected prefix 'cpe', got '{parts[0]}' in '{cpe_str}'"
        )

    if parts[1] != "2.3":
        raise ValueError(
            f"Invalid CPE 2.3 format: expected version '2.3', got '{parts[1]}' in '{cpe_str}'"
        )

    # Extract fields (indices: 0=cpe, 1=2.3, 2=part, 3=vendor, 4=product, 5=version)
    vendor = parts[3]
    product = parts[4]
    version = parts[5]

    return CPEEntry(
        vendor=vendor,
        product=product,
        version=version,
        raw=cpe_str,
    )
