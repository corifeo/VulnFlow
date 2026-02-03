"""Flattening utilities for DataFrame export.

Implements flattening rules from DATA-STRUCTURES.md:
- Most Recent First: Time-series data prefers latest entry
- Authoritative Source First: Official sources (NVD, CISA) over vendor
- Primary/First Entry: For unordered lists, take first item
- Explicit Missing Values: Use None for missing objects
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np


def decode_json_column(col: np.ndarray) -> list[Any]:
    """Decode JSON-encoded column to Python objects.

    Args:
        col: Array of JSON strings (or None/""/null)

    Returns:
        List of decoded Python objects (lists, dicts, None)

    Examples:
        >>> decode_json_column(np.array(['[]', '[{"a": 1}]']))
        [[], [{'a': 1}]]
        >>> decode_json_column(np.array(['null', None, '']))
        [None, None, None]
    """
    result: list[Any] = []
    for item in col:
        # Handle None, empty string, or "null" string
        if item is None or item == "" or item == "null":
            result.append(None)
        else:
            result.append(json.loads(item))
    return result


def flatten_cvss_scores(
    cvss_scores: list[dict[str, Any]] | None,
) -> dict[str, float | str | None]:
    """Flatten CVSS scores list to single entry with source priority.

    Priority order: nvd (0) > redhat (1) > microsoft (2) > others (999)

    Args:
        cvss_scores: List of CVSS dicts with version/vector/base_score/source

    Returns:
        Dict with cvss_version, cvss_vector, cvss_base_score, cvss_source
        (all None if input is None/empty)

    Examples:
        >>> flatten_cvss_scores([{"version": "3.1", "vector": "...", "base_score": 9.8, "source": "nvd"}])
        {'cvss_version': '3.1', 'cvss_vector': '...', 'cvss_base_score': 9.8, 'cvss_source': 'nvd'}
    """
    if not cvss_scores:
        return {
            "cvss_version": None,
            "cvss_vector": None,
            "cvss_base_score": None,
            "cvss_source": None,
        }

    # Source priority map
    source_priority = {"nvd": 0, "redhat": 1, "microsoft": 2}

    # Find entry with highest priority (lowest number)
    best_entry = min(
        cvss_scores,
        key=lambda x: source_priority.get(x.get("source", ""), 999),
    )

    return {
        "cvss_version": best_entry.get("version"),
        "cvss_vector": best_entry.get("vector"),
        "cvss_base_score": best_entry.get("base_score"),
        "cvss_source": best_entry.get("source"),
    }


def flatten_cwes(cwes: list[dict[str, Any]] | None) -> dict[str, str | None]:
    """Flatten CWE list to primary (first) entry.

    Args:
        cwes: List of CWE dicts with id/name

    Returns:
        Dict with cwe_id, cwe_name (both None if input is None/empty)

    Examples:
        >>> flatten_cwes([{"id": "CWE-79", "name": "Cross-site Scripting (XSS)"}])
        {'cwe_id': 'CWE-79', 'cwe_name': 'Cross-site Scripting (XSS)'}
    """
    if not cwes:
        return {"cwe_id": None, "cwe_name": None}

    first_cwe = cwes[0]
    return {
        "cwe_id": first_cwe.get("id"),
        "cwe_name": first_cwe.get("name"),
    }


def flatten_cpes(cpes: list[dict[str, Any]] | None) -> dict[str, str | None]:
    """Flatten CPE list to primary (first) entry.

    Args:
        cpes: List of CPE dicts with uri/vendor/product

    Returns:
        Dict with cpe_uri, cpe_vendor, cpe_product (all None if input is None/empty)

    Examples:
        >>> flatten_cpes([{"uri": "cpe:2.3:a:apache:log4j:...", "vendor": "apache", "product": "log4j"}])
        {'cpe_uri': 'cpe:2.3:a:apache:log4j:...', 'cpe_vendor': 'apache', 'cpe_product': 'log4j'}
    """
    if not cpes:
        return {"cpe_uri": None, "cpe_vendor": None, "cpe_product": None}

    first_cpe = cpes[0]
    return {
        "cpe_uri": first_cpe.get("uri"),
        "cpe_vendor": first_cpe.get("vendor"),
        "cpe_product": first_cpe.get("product"),
    }
