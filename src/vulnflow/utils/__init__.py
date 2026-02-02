"""
Utilities - Shared parsing and lookup utilities

Provides:
- parse_cpe23: CPE 2.3 string parser
- CWEDatabase: CWE database loader

Layer: Utility
Dependencies: schema
"""

from __future__ import annotations

from vulnflow.utils.cpe import parse_cpe23

__all__ = ["parse_cpe23"]
