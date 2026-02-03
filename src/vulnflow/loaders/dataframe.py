"""
DataFrame Loaders - Pandas and Polars export

Provides:
- PandasLoader: Export to pandas DataFrame
- PolarsLoader: Export to polars DataFrame (native, not via pandas)
- to_pandas: Legacy function for backward compatibility
- to_polars: Legacy function for backward compatibility

Layer: Load
Dependencies: dataset
Used by: dataset.to_pandas(), to_polars()
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vulnflow.registry import Registry

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

    from vulnflow.dataset import VulnDataset

# Columns containing nested JSON (can be excluded for cleaner DataFrames)
NESTED_COLUMNS = frozenset({
    "cvss_scores",
    "cwes",
    "cpes",
    "references",
    "exploits",
    "epss_scores",
    "kev",
})


@Registry.register_loader("pandas")
class PandasLoader:
    """Load VulnDataset to pandas DataFrame.

    By default, includes all columns for full data fidelity on round-trip.
    Set exclude_nested=True for cleaner DataFrames (analysis use case).

    Args:
        exclude_nested: If True, exclude nested JSON columns (cvss_scores, etc.)
                       for a cleaner DataFrame. Default False preserves all data.
    """

    format: str = "pandas"

    def __init__(self, exclude_nested: bool = False) -> None:
        """Initialize loader.

        Args:
            exclude_nested: If True, exclude nested JSON columns
        """
        self.exclude_nested = exclude_nested

    def load(self, dataset: VulnDataset) -> pd.DataFrame:
        """Load dataset to pandas DataFrame.

        Args:
            dataset: VulnDataset to export

        Returns:
            Pandas DataFrame
        """
        import pandas as pd

        if self.exclude_nested:
            data = {k: v for k, v in dataset._data.items() if k not in NESTED_COLUMNS}
            return pd.DataFrame(data)

        return pd.DataFrame(dataset._data)


@Registry.register_loader("polars")
class PolarsLoader:
    """Load VulnDataset to polars DataFrame.

    Builds directly from numpy arrays for performance (no pandas intermediate).

    Args:
        exclude_nested: If True, exclude nested JSON columns (cvss_scores, etc.)
                       for a cleaner DataFrame. Default False preserves all data.
    """

    format: str = "polars"

    def __init__(self, exclude_nested: bool = False) -> None:
        """Initialize loader.

        Args:
            exclude_nested: If True, exclude nested JSON columns
        """
        self.exclude_nested = exclude_nested

    def load(self, dataset: VulnDataset) -> pl.DataFrame:
        """Load dataset to polars DataFrame.

        Args:
            dataset: VulnDataset to export

        Returns:
            Polars DataFrame
        """
        import polars as pl

        if self.exclude_nested:
            data = {
                k: v.tolist()
                for k, v in dataset._data.items()
                if k not in NESTED_COLUMNS
            }
        else:
            data = {k: v.tolist() for k, v in dataset._data.items()}

        return pl.from_dict(data)


# ============================================================================
# Legacy Functions (backward compatibility)
# ============================================================================


def to_pandas(dataset: VulnDataset) -> pd.DataFrame:
    """Export dataset to pandas DataFrame.

    LEGACY: Use PandasLoader class for new code.

    Args:
        dataset: VulnDataset to export

    Returns:
        Pandas DataFrame with all columns (no flattening)
    """
    import pandas as pd

    return pd.DataFrame(dataset._data)


def to_polars(dataset: VulnDataset) -> pl.DataFrame:
    """Export dataset to polars DataFrame.

    LEGACY: Use PolarsLoader class for new code.

    Args:
        dataset: VulnDataset to export

    Returns:
        Polars DataFrame with all columns (no flattening)
    """
    import polars as pl

    # Convert numpy arrays to lists for polars compatibility
    data = {k: v.tolist() for k, v in dataset._data.items()}
    return pl.from_dict(data)
