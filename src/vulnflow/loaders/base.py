"""
Loader Base - Protocol for data loading/export

Provides:
- Loader: Protocol defining loader interface

Layer: Load
Dependencies: dataset
Used by: files, dataframe, sql loaders
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from vulnflow.dataset import VulnDataset


@runtime_checkable
class Loader(Protocol):
    """Protocol for data loaders/exporters."""

    def load(self, dataset: VulnDataset) -> dict[str, Any]:
        """Load/export dataset to destination.

        Returns:
            Metadata dict with loader-specific information
        """
        ...
