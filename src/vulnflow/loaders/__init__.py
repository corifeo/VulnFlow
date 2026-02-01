"""
Loaders - Data output writers

Provides:
- Loader: Protocol for data loading/export
- ParquetLoader: Parquet file writer
- DataFrameLoader: Pandas/Polars export

Layer: Load
Dependencies: schema, dataset
Used by: dataset.to_parquet(), to_pandas(), to_polars()
"""

from vulnflow.loaders.base import Loader

__all__ = ["Loader"]
