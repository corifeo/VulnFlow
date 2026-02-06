"""Utilities - Shared parsing and lookup utilities.

Provides:
- parse_cpe23: CPE 2.3 string parser
- topological_sort: Dependency resolution for transforms
- get_record_columns: Get VulnRecord field names
- get_flattened_columns: Get flattened column names
- is_async_transform: Check if object has async transform() method
- partition_transforms: Separate async from sync transforms
- is_url: Check if source is HTTP/HTTPS URL
- detect_format: Detect content format from URL, headers, or content
- HttpError: HTTP exception wrapper (prevents httpx leakage)
- HttpFetcher: Async HTTP client with disk caching and gzip support
- progress_bar: Factory function for progress bars with VulnFlow defaults
- normalize_source: Normalize CVSS source strings to canonical form

Layer: Utility
Dependencies: schema
"""

from __future__ import annotations

from vulnflow.utils import json
from vulnflow.utils.cpe import parse_cpe23
from vulnflow.utils.cvss import normalize_source
from vulnflow.utils.graph import topological_sort
from vulnflow.utils.http import (
    ContentFormat,
    HttpError,
    HttpFetcher,
    detect_format,
    is_url,
)
from vulnflow.utils.progress import (
    ProgressReporter,
    progress_bar,
)
from vulnflow.utils.schema import get_flattened_columns, get_record_columns
from vulnflow.utils.transforms import is_async_transform, partition_transforms

__all__ = [
    "parse_cpe23",
    "normalize_source",
    "topological_sort",
    "get_record_columns",
    "get_flattened_columns",
    "is_async_transform",
    "partition_transforms",
    "is_url",
    "detect_format",
    "HttpError",
    "HttpFetcher",
    "ContentFormat",
    "ProgressReporter",
    "progress_bar",
    "json",
]
