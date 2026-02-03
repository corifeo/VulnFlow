"""Utilities - Shared parsing and lookup utilities.

Provides:
- parse_cpe23: CPE 2.3 string parser
- CWEDatabase: CWE database loader
- topological_sort: Dependency resolution for transforms
- get_record_columns: Get VulnRecord field names
- get_flattened_columns: Get flattened column names
- is_async_transform: Check if object has async transform() method
- partition_transforms: Separate async from sync transforms

Layer: Utility
Dependencies: schema
"""

from __future__ import annotations

from vulnflow.utils.cpe import parse_cpe23
from vulnflow.utils.cwe import CWEDatabase
from vulnflow.utils.graph import topological_sort
from vulnflow.utils.schema import get_flattened_columns, get_record_columns
from vulnflow.utils.transforms import is_async_transform, partition_transforms

__all__ = [
    "parse_cpe23",
    "CWEDatabase",
    "topological_sort",
    "get_record_columns",
    "get_flattened_columns",
    "is_async_transform",
    "partition_transforms",
]
