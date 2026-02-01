"""
DataFrame Loaders - Pandas and Polars export

Provides:
- to_pandas: Export dataset to pandas DataFrame
- to_polars: Export dataset to polars DataFrame

Layer: Load
Dependencies: dataset, schema
Used by: dataset.to_pandas(), to_polars()
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vulnflow.dataset import VulnDataset


def to_pandas(dataset: VulnDataset) -> Any:
    """Export dataset to pandas DataFrame."""
    raise NotImplementedError("Pandas export not yet implemented")


def to_polars(dataset: VulnDataset) -> Any:
    """Export dataset to polars DataFrame."""
    raise NotImplementedError("Polars export not yet implemented")
