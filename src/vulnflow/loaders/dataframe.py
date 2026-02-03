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


def to_pandas(dataset: VulnDataset, use_registry: bool = False) -> Any:
    """Export dataset to pandas DataFrame.

    Args:
        dataset: VulnDataset to export
        use_registry: If True, only export registered schema fields

    Returns:
        pandas DataFrame
    """
    import pandas as pd

    if use_registry:
        from vulnflow.schema_registry import get_global_registry

        registry = get_global_registry()
        registered_fields = registry.get_registered_fields()

        # Filter to registered fields only
        filtered_data = {
            name: dataset._data[name] for name in registered_fields if name in dataset._data
        }
        return pd.DataFrame(filtered_data)
    else:
        # Export all fields
        return pd.DataFrame(dataset._data)


def to_polars(dataset: VulnDataset, use_registry: bool = False) -> Any:
    """Export dataset to polars DataFrame.

    Args:
        dataset: VulnDataset to export
        use_registry: If True, only export registered schema fields

    Returns:
        polars DataFrame
    """
    import polars as pl

    if use_registry:
        from vulnflow.schema_registry import get_global_registry

        registry = get_global_registry()
        registered_fields = registry.get_registered_fields()

        # Filter to registered fields only
        filtered_data = {
            name: dataset._data[name].tolist()
            for name in registered_fields
            if name in dataset._data
        }
        return pl.from_dict(filtered_data)
    else:
        # Export all fields
        data_as_lists = {k: v.tolist() for k, v in dataset._data.items()}
        return pl.from_dict(data_as_lists)
