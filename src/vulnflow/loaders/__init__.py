"""
Loaders - Data output writers

Provides:
- Loader: Protocol for data loading/export
- BaseLoader: ABC for loaders with format validation
- ParquetLoader, JSONLoader: File writers
- PandasLoader, PolarsLoader: DataFrame export
- TriageLoader: Ranked SSVC triage queue (JSON) for VulnTriage

Layer: Load
Dependencies: schema, dataset
Used by: dataset.to_parquet(), to_pandas(), to_polars()
"""

from vulnflow.loaders.base import BaseLoader, Loader
from vulnflow.loaders.dataframe import PandasLoader, PolarsLoader
from vulnflow.loaders.files import JSONLoader, ParquetLoader
from vulnflow.loaders.triage import TRIAGE_SCHEMA, TriageLoader

__all__ = [
    "Loader",
    "BaseLoader",
    "ParquetLoader",
    "JSONLoader",
    "PandasLoader",
    "PolarsLoader",
    "TriageLoader",
    "TRIAGE_SCHEMA",
]
