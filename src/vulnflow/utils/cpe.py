"""
CPE Parser - Parse CPE 2.3 formatted strings

Provides:
- parse_cpe23: Parse CPE 2.3 string into CPEEntry

Layer: Utility
Dependencies: schema
"""

from __future__ import annotations

from vulnflow.schema import CPEEntry


def parse_cpe23(cpe_str: str) -> CPEEntry:
    """Parse CPE 2.3 formatted string into structured CPEEntry.

    CPE 2.3 format: cpe:2.3:part:vendor:product:version:update:edition:language:sw_edition:target_sw:target_hw:other

    Args:
        cpe_str: CPE 2.3 formatted string

    Returns:
        CPEEntry with parsed vendor, product, version

    Raises:
        ValueError: If CPE format is invalid
    """
    if not cpe_str:
        raise ValueError("CPE string cannot be empty")

    parts = cpe_str.split(":")

    # Valid CPE 2.3: cpe:2.3:part:vendor:product:version:...
    # Minimum 6 parts (up to version field)
    if len(parts) < 6 or parts[0] != "cpe" or parts[1] != "2.3":
        raise ValueError(f"Invalid CPE 2.3 format: {cpe_str}")

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
