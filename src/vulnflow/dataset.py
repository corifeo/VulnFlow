"""
VulnDataset - Core container for vulnerability records

Provides:
- VulnDataset: Main class for ETL operations on vulnerability data

Layer: Core
Dependencies: schema, registry
Used by: __init__, user code
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vulnflow.schema import VulnRecord


class VulnDataset:
    """Core container for vulnerability records with ETL operations."""

    def __init__(self) -> None:
        self._records: list[VulnRecord] = []

    def __len__(self) -> int:
        return len(self._records)

    @classmethod
    def _from_records(cls, records: list[VulnRecord]) -> VulnDataset:
        """Create dataset from list of VulnRecords (internal factory)."""
        dataset = cls()
        dataset._records = records
        return dataset
