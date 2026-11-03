"""Utilities - Shared parsing and lookup utilities.

Provides:
- parse_cpe23: CPE 2.3 string parser
- topological_sort: Dependency resolution for transforms
- get_record_columns: Get VulnRecord field names
- get_flattened_columns: Get flattened column names
- is_async_transform: Check if object has async transform() method
- is_url: Check if source is HTTP/HTTPS URL
- detect_format: Detect content format from URL, headers, or content
- HttpError: HTTP exception wrapper (prevents httpx leakage)
- HttpFetcher: Async HTTP client with disk caching and gzip support
- progress_bar: Factory function for progress bars with VulnFlow defaults
- normalize_source: Normalize CVSS source strings to canonical form
- score_to_severity: Map CVSS base score to severity string
- parse_timestamp: ISO timestamp string to datetime
- to_utc_datetime: Any timestamp form to timezone-aware UTC datetime
- extract_nvd_status: Normalize NVD vulnStatus field
- parse_cvss: Parse CVSS v2/v3.0/v3.1/v4.0 metrics
- parse_cwes: Parse CWE weaknesses
- parse_cpes: Parse CPE configurations
- parse_references: Parse reference entries
- populate_columns: Convert NVD vulnerability dicts to columnar arrays

Layer: Utility
Dependencies: schema
"""

from __future__ import annotations

from vulnflow.utils import json
from vulnflow.utils.cpe import parse_cpe23
from vulnflow.utils.cvss import normalize_source, score_to_severity
from vulnflow.utils.graph import topological_sort
from vulnflow.utils.http import (
    ContentFormat,
    HttpError,
    HttpFetcher,
    detect_format,
    is_url,
)
from vulnflow.utils.nvd import (
    extract_nvd_status,
    parse_cpes,
    parse_cvss,
    parse_cwes,
    parse_references,
    parse_timestamp,
    populate_columns,
    to_utc_datetime,
)
from vulnflow.utils.progress import (
    ProgressMode,
    ProgressReporter,
    progress_bar,
)
from vulnflow.utils.schema import get_flattened_columns, get_record_columns
from vulnflow.utils.transforms import is_async_transform

__all__ = [
    "parse_cpe23",
    "normalize_source",
    "score_to_severity",
    "parse_timestamp",
    "to_utc_datetime",
    "extract_nvd_status",
    "parse_cvss",
    "parse_cwes",
    "parse_cpes",
    "parse_references",
    "populate_columns",
    "topological_sort",
    "get_record_columns",
    "get_flattened_columns",
    "is_async_transform",
    "is_url",
    "detect_format",
    "HttpError",
    "HttpFetcher",
    "ContentFormat",
    "ProgressMode",
    "ProgressReporter",
    "progress_bar",
    "json",
]
