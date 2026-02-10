"""
DataFrame Loaders - Pandas and Polars export

Provides:
- PandasLoader: Export to pandas DataFrame (with NVD status filtering)
- PolarsLoader: Export to polars DataFrame (with NVD status filtering)
- to_pandas: Legacy function for backward compatibility
- to_polars: Legacy function for backward compatibility

DataFrame exports are the *analysis path* — they filter to sane defaults.
File exports (parquet, json) are the *archive path* — always lossless.

Layer: Load
Dependencies: dataset
Used by: dataset.to_pandas(), to_polars()
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from vulnflow.registry import Registry
from vulnflow.utils.filtering import (
    _NVD_STATUS_DEFAULTS,
    _build_date_mask,
    _build_status_mask,
)

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

    from vulnflow.dataset import VulnDataset

# Columns containing nested JSON (can be excluded for cleaner DataFrames)
NESTED_COLUMNS = frozenset(
    {
        "cvss_scores",
        "cwes",
        "cpes",
        "references",
        "exploits",
        "epss_scores",
        "kev",
    }
)


# ============================================================================
# DataFrame Loaders
# ============================================================================


@Registry.register_loader("pandas")
class PandasLoader:
    """Load VulnDataset to pandas DataFrame.

    By default, excludes nested JSON columns for cleaner DataFrames (lossy).
    Set include_nested=True for full data fidelity on round-trip.

    Filters rows by NVD status (analysis path). Default excludes only
    rejected rows.
    """

    format: str = "pandas"

    def __init__(
        self,
        include_nested: bool = False,
        *,
        start: str | datetime | None = None,
        end: str | datetime | None = None,
        **status_kwargs: bool,
    ) -> None:
        """Initialize loader.

        Args:
            include_nested: If True, include nested JSON columns for full fidelity.
                           Default False excludes nested columns (lossy by default).
            start: Inclusive lower bound (>=) for nvd_published_at. ISO string or datetime.
            end: Exclusive upper bound (<) for nvd_published_at. ISO string or datetime.
            **status_kwargs: Override NVD status filters (analyzed, modified,
                            awaiting, rejected, deferred). Default excludes
                            only rejected rows.
        """
        self.include_nested = include_nested
        self._start = start
        self._end = end
        self._status_kwargs = {
            k: status_kwargs.get(k, v) for k, v in _NVD_STATUS_DEFAULTS.items()
        }

    def load(self, dataset: VulnDataset) -> pd.DataFrame:
        """Load dataset to pandas DataFrame.

        Args:
            dataset: VulnDataset to export

        Returns:
            Pandas DataFrame
        """
        import pandas as pd

        mask = _build_status_mask(dataset, **self._status_kwargs) & _build_date_mask(
            dataset, start=self._start, end=self._end
        )
        if not self.include_nested:
            data = {
                k: v[mask] for k, v in dataset._data.items() if k not in NESTED_COLUMNS
            }
            return pd.DataFrame(data)

        return pd.DataFrame({k: v[mask] for k, v in dataset._data.items()})


@Registry.register_loader("polars")
class PolarsLoader:
    """Load VulnDataset to polars DataFrame.

    Builds directly from numpy arrays for performance (no pandas intermediate).

    By default, excludes nested JSON columns for cleaner DataFrames (lossy).
    Set include_nested=True for full data fidelity on round-trip.

    Filters rows by NVD status (analysis path). Default excludes only
    rejected rows.
    """

    format: str = "polars"

    def __init__(
        self,
        include_nested: bool = False,
        *,
        start: str | datetime | None = None,
        end: str | datetime | None = None,
        **status_kwargs: bool,
    ) -> None:
        """Initialize loader.

        Args:
            include_nested: If True, include nested JSON columns for full fidelity.
                           Default False excludes nested columns (lossy by default).
            start: Inclusive lower bound (>=) for nvd_published_at. ISO string or datetime.
            end: Exclusive upper bound (<) for nvd_published_at. ISO string or datetime.
            **status_kwargs: Override NVD status filters (analyzed, modified,
                            awaiting, rejected, deferred). Default excludes
                            only rejected rows.
        """
        self.include_nested = include_nested
        self._start = start
        self._end = end
        self._status_kwargs = {
            k: status_kwargs.get(k, v) for k, v in _NVD_STATUS_DEFAULTS.items()
        }

    def load(self, dataset: VulnDataset) -> pl.DataFrame:
        """Load dataset to polars DataFrame.

        Args:
            dataset: VulnDataset to export

        Returns:
            Polars DataFrame
        """
        import polars as pl

        mask = _build_status_mask(dataset, **self._status_kwargs) & _build_date_mask(
            dataset, start=self._start, end=self._end
        )
        if not self.include_nested:
            data = {
                k: v[mask].tolist()
                for k, v in dataset._data.items()
                if k not in NESTED_COLUMNS
            }
        else:
            data = {k: v[mask].tolist() for k, v in dataset._data.items()}

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
