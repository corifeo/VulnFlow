"""
File Extractors - Parquet, JSON, CSV readers

Provides:
- ParquetExtractor: Reads from Parquet files
- JSONExtractor: Reads from JSON files (schema-aware round-trip)
- PandasExtractor: Reads from pandas DataFrames
- PolarsExtractor: Reads from Polars DataFrames

Layer: Extract
Dependencies: None
Used by: dataset.from_parquet(), from_json(), from_pandas(), from_polars()
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID

import numpy as np

from vulnflow.extractors.base import BaseExtractor
from vulnflow.registry import Registry

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

    from vulnflow.dataset import VulnDataset

logger = logging.getLogger(__name__)


@Registry.register_extractor("parquet")
class ParquetExtractor(BaseExtractor):
    """Extract from Parquet files (direct load).

    Parquet is already columnar, so we load directly to VulnDataset
    without intermediate VulnRecord conversion.
    """

    name: str = "parquet"
    format: str = "parquet"

    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = path

    def extract(self) -> VulnDataset:
        """Extract from Parquet → VulnDataset directly.

        Returns:
            VulnDataset with columnar data from Parquet file

        Raises:
            FileNotFoundError: If Parquet file doesn't exist
        """
        import pyarrow.parquet as pq

        from vulnflow.dataset import VulnDataset

        path = Path(self.path)
        if not path.exists():
            raise FileNotFoundError(f"Parquet file not found: {path}")

        table = pq.read_table(path)

        dataset = VulnDataset()
        dataset._data = {col: table[col].to_numpy() for col in table.column_names}

        return dataset


@Registry.register_extractor("json")
class JSONExtractor(BaseExtractor):
    """Extract from JSON files (schema-aware round-trip from JSONLoader).

    Uses Registry schema to reconstruct accurate types:
    - datetime, UUID: Proper reconstruction (not lossy!)
    - int, float, bool: Typed conversion
    - JSON-encoded columns: Preserved as strings

    For maximum performance, use ParquetExtractor.
    """

    name: str = "json"
    format: str = "json"

    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = path

    def extract(self) -> VulnDataset:
        """Extract from JSON → VulnDataset with schema-aware type reconstruction.

        Returns:
            VulnDataset with schema-aware type conversion

        Raises:
            FileNotFoundError: If JSON file doesn't exist
            ValueError: If JSON is not a list
        """
        from vulnflow.dataset import VulnDataset

        path = Path(self.path)
        self._ensure_file_exists(path)

        data = self._read_json_safely(path)

        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array, got {type(data).__name__}")

        if not data:
            return VulnDataset()

        # Build type map from Registry
        type_map = self._build_type_map()

        # Extract all column names
        all_columns: set[str] = set()
        for record in data:
            all_columns.update(record.keys())

        # Build columnar data
        columnar_data: dict[str, np.ndarray] = {}

        for col in all_columns:
            # Collect all values for this column
            values = []
            for record in data:
                value = record.get(col)
                converted = self._convert_value(value, col, type_map)
                values.append(converted)

            # Create numpy array with object dtype to handle heterogeneous types
            columnar_data[col] = np.array(values, dtype=object)

        dataset = VulnDataset()
        dataset._data = columnar_data

        # Warn if core fields missing
        if "cve_id" not in all_columns:
            logger.warning("JSON dataset missing 'cve_id' column")
        if "description" not in all_columns:
            logger.warning("JSON dataset missing 'description' column")

        return dataset

    def _build_type_map(self) -> dict[str, str]:
        """Build mapping of column name → type from Registry.

        Returns:
            Dictionary mapping column names to their expected types
        """
        type_map = {}
        for field_def in Registry.get_all_columns():
            type_map[field_def.name] = field_def.type
        return type_map

    def _convert_value(self, value: Any, column: str, type_map: dict[str, str]) -> Any:
        """Convert value using schema type information.

        Args:
            value: Value to convert
            column: Column name (for schema lookup)
            type_map: Mapping of column names to types

        Returns:
            Converted value with appropriate type
        """
        if value is None or value == "None":
            return None

        # Get expected type from schema
        expected_type = type_map.get(column)

        if expected_type == "datetime":
            return datetime.fromisoformat(value)
        elif expected_type == "uuid":
            return UUID(value)
        elif expected_type == "int":
            return int(value)
        elif expected_type == "float":
            return float(value)
        elif expected_type == "bool":
            return value == "True"
        elif expected_type == "str":
            return value
        else:
            # Fallback: basic inference for unregistered fields
            return self._infer_type(value)

    def _infer_type(self, value: Any) -> Any:
        """Fallback type inference for unregistered fields.

        Args:
            value: Value to infer type for

        Returns:
            Value converted to inferred type
        """
        if isinstance(value, str):
            # Try int
            if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
                return int(value)
            # Try float
            try:
                return float(value)
            except ValueError:
                pass
            # Try bool
            if value in ("True", "False"):
                return value == "True"

        # Keep as-is
        return value


@Registry.register_extractor("pandas")
class PandasExtractor(BaseExtractor):
    """Extract from pandas DataFrame (in-memory).

    Accepts flattened DataFrames (scalar columns only).
    Does NOT reconstruct nested structures.
    """

    name: str = "pandas"
    format: str = "pandas"

    def __init__(self, dataframe: pd.DataFrame) -> None:
        super().__init__()
        self.dataframe = dataframe

    def extract(self) -> VulnDataset:
        """Extract from DataFrame → VulnDataset.

        Returns:
            VulnDataset with columnar data from DataFrame
        """
        from vulnflow.dataset import VulnDataset

        if self.dataframe.empty:
            return VulnDataset()

        # Convert columns to numpy arrays
        dataset = VulnDataset()
        dataset._data = {
            col: self.dataframe[col].to_numpy() for col in self.dataframe.columns
        }

        # Warn if core fields missing
        if "cve_id" not in self.dataframe.columns:
            logger.warning("DataFrame missing 'cve_id' column")
        if "description" not in self.dataframe.columns:
            logger.warning("DataFrame missing 'description' column")

        return dataset


@Registry.register_extractor("polars")
class PolarsExtractor(BaseExtractor):
    """Extract from Polars DataFrame (delegates to PandasExtractor)."""

    name: str = "polars"
    format: str = "polars"

    def __init__(self, dataframe: pl.DataFrame) -> None:
        super().__init__()
        self.dataframe = dataframe

    def extract(self) -> VulnDataset:
        """Extract from Polars → VulnDataset via Pandas.

        Returns:
            VulnDataset with columnar data from Polars DataFrame
        """
        pandas_df = self.dataframe.to_pandas()
        extractor = PandasExtractor(pandas_df)
        return extractor.extract()
