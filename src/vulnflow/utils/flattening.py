"""Flattening utilities for DataFrame export.

Provides:
- decode_json_column: Decode JSON-encoded column to Python objects
- flatten_cvss_scores: Flatten list[CVSSEntry] to scalar columns
- flatten_cwes: Flatten CWE list to primary entry
- flatten_cpes: Flatten CPE list to primary entry

Layer: Utility
Dependencies: None
Used by: loaders/dataframe

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

    Handles None, empty strings, and "null" strings by returning None.
    All other values are parsed as JSON.

    Args:
        col: NumPy array of JSON strings (or None/""/null)

    Returns:
        List of decoded Python objects (lists, dicts, or None)

    Raises:
        json.JSONDecodeError: If any non-null value is invalid JSON

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
    """Flatten list[CVSSEntry] to scalar columns.

    Selection logic (per DATA-STRUCTURES.md):
    1. Prefer source: nvd > redhat > microsoft > other
    2. If tie, prefer most recent published_at
    3. If tie, take first

    Args:
        cvss_scores: List of CVSS score dicts (decoded from JSON), or None

    Returns:
        Dict with keys: cvss_base_score, cvss_base_severity, cvss_version,
        cvss_vector_string, cvss_source. All values None if input is None/empty
        or no valid scores found.

    Examples:
        >>> scores = [{
        ...     "version": "3.1",
        ...     "vector_string": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
        ...     "base_score": 9.8,
        ...     "base_severity": "CRITICAL",
        ...     "source": "nvd"
        ... }]
        >>> flatten_cvss_scores(scores)
        {'cvss_base_score': 9.8, 'cvss_base_severity': 'CRITICAL', ...}
        >>> flatten_cvss_scores(None)
        {'cvss_base_score': None, 'cvss_base_severity': None, ...}
    """
    if not cvss_scores:
        return {
            "cvss_base_score": None,
            "cvss_base_severity": None,
            "cvss_version": None,
            "cvss_vector_string": None,
            "cvss_source": None,
        }

    # Filter entries with valid base_score
    valid_scores = [
        entry for entry in cvss_scores if entry.get("base_score") is not None
    ]

    if not valid_scores:
        return {
            "cvss_base_score": None,
            "cvss_base_severity": None,
            "cvss_version": None,
            "cvss_vector_string": None,
            "cvss_source": None,
        }

    # Source preference (NVD domain knowledge)
    source_priority = {"nvd": 0, "redhat": 1, "microsoft": 2}

    def score_key(entry: dict[str, Any]) -> tuple[int, float]:
        source_rank = source_priority.get(entry.get("source", ""), 999)
        # Negative timestamp for descending sort (most recent first)
        time_rank = -entry.get("published_at", 0) if entry.get("published_at") else 0
        return (source_rank, time_rank)

    selected = min(valid_scores, key=score_key)

    return {
        "cvss_base_score": selected.get("base_score"),
        "cvss_base_severity": selected.get("base_severity"),
        "cvss_version": selected.get("version"),
        "cvss_vector_string": selected.get("vector_string"),
        "cvss_source": selected.get("source"),
    }


def flatten_cwes(cwes: list[dict[str, Any]] | None) -> dict[str, str | None]:
    """Flatten CWE list to primary (first) entry.

    Applies "Primary/First Entry" rule from DATA-STRUCTURES.md.

    Args:
        cwes: List of CWE dicts with id/name fields, or None

    Returns:
        Dict with keys: primary_cwe_id, primary_cwe_name.
        Both values None if input is None/empty.

    Examples:
        >>> flatten_cwes([{"id": "CWE-79", "name": "Cross-site Scripting (XSS)"}])
        {'primary_cwe_id': 'CWE-79', 'primary_cwe_name': 'Cross-site Scripting (XSS)'}
        >>> flatten_cwes(None)
        {'primary_cwe_id': None, 'primary_cwe_name': None}
    """
    if not cwes:
        return {"primary_cwe_id": None, "primary_cwe_name": None}

    first_cwe = cwes[0]
    return {
        "primary_cwe_id": first_cwe.get("id"),
        "primary_cwe_name": first_cwe.get("name"),
    }


def flatten_cpes(cpes: list[dict[str, Any]] | None) -> dict[str, str | None]:
    """Flatten CPE list to primary (first) entry.

    Applies "Primary/First Entry" rule from DATA-STRUCTURES.md.

    Args:
        cpes: List of CPE dicts with uri/vendor/product/version fields, or None

    Returns:
        Dict with keys: primary_vendor, primary_product, primary_version.
        All values None if input is None/empty.

    Examples:
        >>> cpes = [{"vendor": "apache", "product": "log4j", "version": "2.14.1"}]
        >>> flatten_cpes(cpes)
        {'primary_vendor': 'apache', 'primary_product': 'log4j', 'primary_version': '2.14.1'}
        >>> flatten_cpes(None)
        {'primary_vendor': None, 'primary_product': None, 'primary_version': None}
    """
    if not cpes:
        return {
            "primary_vendor": None,
            "primary_product": None,
            "primary_version": None,
        }

    first_cpe = cpes[0]
    return {
        "primary_vendor": first_cpe.get("vendor"),
        "primary_product": first_cpe.get("product"),
        "primary_version": first_cpe.get("version"),
    }
