"""
Extractor Base - Protocol for data extraction

Provides:
- Extractor: Protocol defining extractor interface

Layer: Extract
Dependencies: schema
Used by: nvd, files, sql, memory extractors
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from vulnflow.schema import VulnRecord


@runtime_checkable
class Extractor(Protocol):
    """Protocol for data extractors."""

    def extract(self) -> list[VulnRecord]:
        """Extract vulnerability records from source."""
        ...
