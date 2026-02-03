"""
DataFrame Loaders - Pandas and Polars export

Provides:
- PandasLoader: Class-based loader with schema-aware flattening
- PolarsLoader: Class-based loader delegating to PandasLoader
- to_pandas: Legacy function for backward compatibility
- to_polars: Legacy function for backward compatibility

Layer: Load
Dependencies: dataset, schema, utils.flattening
Used by: dataset.to_pandas(), to_polars()
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

    from vulnflow.dataset import VulnDataset


class PandasLoader:
    """Load VulnDataset to pandas DataFrame with schema-aware flattening.

    Flattens nested JSON columns according to DATA-STRUCTURES.md rules:
    - cvss_scores → cvss_base_score, cvss_base_severity, cvss_version, etc.
    - cwes → primary_cwe_id, primary_cwe_name
    - cpes → primary_vendor, primary_product, primary_version

    Args:
        include_nested: If True, keep raw JSON columns in addition to flattened columns.
                       If False (default), only include flattened scalar columns.
    """

    def __init__(self, include_nested: bool = False) -> None:
        """Initialize loader.

        Args:
            include_nested: If True, keep both flattened and raw JSON columns
        """
        self.include_nested = include_nested

    def load(self, dataset: VulnDataset) -> pd.DataFrame:
        """Load dataset to pandas DataFrame.

        Args:
            dataset: VulnDataset to export

        Returns:
            Pandas DataFrame with flattened columns
        """
        import pandas as pd

        from vulnflow.utils.flattening import (
            decode_json_column,
            flatten_cpes,
            flatten_cvss_scores,
            flatten_cwes,
        )

        # Start with all columnar data
        data = dict(dataset._data)

        # Columns to skip unless include_nested=True
        skip_unless_nested = {
            "references",
            "exploits",
            "epss_scores",
            "kev",
        }

        # Decode and flatten cvss_scores
        if "cvss_scores" in data:
            cvss_decoded = decode_json_column(data["cvss_scores"])
            flattened_rows_cvss: list[dict[str, float | str | None]] = [flatten_cvss_scores(row) for row in cvss_decoded]

            # Add flattened columns
            for key in [
                "cvss_base_score",
                "cvss_base_severity",
                "cvss_version",
                "cvss_vector_string",
                "cvss_source",
            ]:
                data[key] = np.array([row[key] for row in flattened_rows_cvss], dtype=object)

            # Remove raw column unless include_nested
            if not self.include_nested:
                del data["cvss_scores"]

        # Decode and flatten cwes
        if "cwes" in data:
            cwes_decoded = decode_json_column(data["cwes"])
            flattened_rows_cwe: list[dict[str, str | None]] = [flatten_cwes(row) for row in cwes_decoded]

            # Add flattened columns
            for key in ["primary_cwe_id", "primary_cwe_name"]:
                data[key] = np.array([row[key] for row in flattened_rows_cwe], dtype=object)

            # Remove raw column unless include_nested
            if not self.include_nested:
                del data["cwes"]

        # Decode and flatten cpes
        if "cpes" in data:
            cpes_decoded = decode_json_column(data["cpes"])
            flattened_rows_cpe: list[dict[str, str | None]] = [flatten_cpes(row) for row in cpes_decoded]

            # Add flattened columns
            for key in ["primary_vendor", "primary_product", "primary_version"]:
                data[key] = np.array([row[key] for row in flattened_rows_cpe], dtype=object)

            # Remove raw column unless include_nested
            if not self.include_nested:
                del data["cpes"]

        # Remove skip_unless_nested columns if include_nested=False
        if not self.include_nested:
            for col in skip_unless_nested:
                if col in data:
                    del data[col]

        return pd.DataFrame(data)


class PolarsLoader:
    """Load VulnDataset to polars DataFrame with schema-aware flattening.

    Delegates to PandasLoader and converts result to Polars.

    Args:
        include_nested: If True, keep raw JSON columns in addition to flattened columns.
                       If False (default), only include flattened scalar columns.
    """

    def __init__(self, include_nested: bool = False) -> None:
        """Initialize loader.

        Args:
            include_nested: If True, keep both flattened and raw JSON columns
        """
        self.include_nested = include_nested

    def load(self, dataset: VulnDataset) -> pl.DataFrame:
        """Load dataset to polars DataFrame.

        Args:
            dataset: VulnDataset to export

        Returns:
            Polars DataFrame with flattened columns
        """
        import polars as pl

        # Delegate to PandasLoader and convert
        pandas_loader = PandasLoader(include_nested=self.include_nested)
        pdf = pandas_loader.load(dataset)
        return pl.from_pandas(pdf)


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
