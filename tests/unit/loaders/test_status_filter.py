"""
Status Filters - NVD status filtering for DataFrame loaders only

Tests: _build_status_mask, _NVD_STATUS_DEFAULTS, DataFrame loader integration
File loaders (parquet, json) are lossless — no status filtering.

Layer: Load
Dependencies: dataset, loaders
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from vulnflow.dataset import VulnDataset
from vulnflow.utils.filtering import _NVD_STATUS_DEFAULTS, _build_status_mask

# ============================================================================
# Fixture helper
# ============================================================================


def _dataset_with_statuses(statuses: list[str]) -> VulnDataset:
    """Build a minimal VulnDataset with given nvd_status values."""
    ds = VulnDataset()
    n = len(statuses)
    ds.set_column(
        "cve_id", np.array([f"CVE-2024-{i:04d}" for i in range(n)], dtype=object)
    )
    ds.set_column("nvd_status", np.array(statuses, dtype=object))
    return ds


# ============================================================================
# _build_status_mask unit tests
# ============================================================================


def test_defaults_dict_has_all_five_statuses() -> None:
    """_NVD_STATUS_DEFAULTS has exactly the 5 official NVD statuses."""
    assert set(_NVD_STATUS_DEFAULTS) == {
        "analyzed",
        "modified",
        "awaiting",
        "rejected",
        "deferred",
    }


def test_defaults_include_analyzed_modified() -> None:
    """Default includes analyzed, modified."""
    assert _NVD_STATUS_DEFAULTS["analyzed"] is True
    assert _NVD_STATUS_DEFAULTS["modified"] is True


def test_defaults_exclude_only_rejected() -> None:
    """Default excludes only rejected."""
    assert _NVD_STATUS_DEFAULTS["awaiting"] is True
    assert _NVD_STATUS_DEFAULTS["rejected"] is False
    assert _NVD_STATUS_DEFAULTS["deferred"] is True


def test_mask_with_defaults() -> None:
    """_build_status_mask with defaults keeps all except rejected."""
    ds = _dataset_with_statuses(
        ["analyzed", "awaiting", "modified", "rejected", "deferred"]
    )
    mask = _build_status_mask(ds)
    assert mask.tolist() == [True, True, True, False, True]


def test_mask_include_all() -> None:
    """Passing rejected=True keeps everything."""
    ds = _dataset_with_statuses(["analyzed", "rejected", "deferred"])
    mask = _build_status_mask(ds, rejected=True)
    assert mask.all()


def test_mask_exclude_all() -> None:
    """Passing all=False drops everything."""
    ds = _dataset_with_statuses(["analyzed", "modified"])
    mask = _build_status_mask(ds, analyzed=False, modified=False)
    assert not mask.any()


def test_mask_no_nvd_status_column() -> None:
    """Dataset without nvd_status -> all-True mask (no-op)."""
    ds = VulnDataset()
    ds.set_column("cve_id", np.array(["CVE-2024-0001", "CVE-2024-0002"], dtype=object))
    mask = _build_status_mask(ds)
    assert mask.dtype == bool
    assert mask.all()
    assert len(mask) == 2


def test_mask_empty_dataset() -> None:
    """Empty dataset -> empty boolean mask."""
    ds = VulnDataset()
    mask = _build_status_mask(ds)
    assert mask.dtype == bool
    assert len(mask) == 0


def test_mask_dtype_is_bool() -> None:
    """Mask dtype is always bool."""
    ds = _dataset_with_statuses(["analyzed"])
    mask = _build_status_mask(ds)
    assert mask.dtype == bool


def test_mask_single_override() -> None:
    """Override a single kwarg to exclude a default-included status."""
    ds = _dataset_with_statuses(["awaiting", "analyzed"])
    mask = _build_status_mask(ds, awaiting=False)
    assert mask.tolist() == [False, True]


def test_mask_none_status_always_included() -> None:
    """Rows with None status are always included."""
    ds = VulnDataset()
    ds.set_column(
        "cve_id",
        np.array(["CVE-2016-0001", "CVE-2024-0001", "CVE-2024-0002"], dtype=object),
    )
    ds.set_column("nvd_status", np.array([None, "analyzed", "rejected"], dtype=object))
    mask = _build_status_mask(ds)
    # None → included, analyzed → included, rejected → excluded
    assert mask.tolist() == [True, True, False]


def test_mask_all_none_statuses() -> None:
    """Column exists but all values are None → all included."""
    ds = VulnDataset()
    ds.set_column("cve_id", np.array(["CVE-2015-0001", "CVE-2016-0001"], dtype=object))
    ds.set_column("nvd_status", np.array([None, None], dtype=object))
    mask = _build_status_mask(ds)
    assert mask.all()


# ============================================================================
# PandasLoader integration
# ============================================================================


def test_pandas_loader_filters_by_default() -> None:
    """PandasLoader with defaults keeps all except rejected."""
    ds = _dataset_with_statuses(["analyzed", "awaiting", "modified", "rejected"])
    df = ds.to_pandas()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert "rejected" not in df["nvd_status"].tolist()


def test_pandas_loader_include_all_statuses() -> None:
    """PandasLoader with all statuses enabled keeps all rows."""
    ds = _dataset_with_statuses(["analyzed", "awaiting", "rejected"])
    df = ds.to_pandas(rejected=True)
    assert len(df) == 3


def test_pandas_loader_no_nvd_status_keeps_all() -> None:
    """PandasLoader without nvd_status column keeps all rows."""
    ds = VulnDataset()
    ds.set_column("cve_id", np.array(["CVE-2024-0001"], dtype=object))
    df = ds.to_pandas()
    assert len(df) == 1


# ============================================================================
# PolarsLoader integration
# ============================================================================


def test_polars_loader_filters_by_default() -> None:
    """PolarsLoader with defaults keeps all except rejected."""
    ds = _dataset_with_statuses(["analyzed", "awaiting", "modified", "rejected"])
    df = ds.to_polars()
    assert isinstance(df, pl.DataFrame)
    assert len(df) == 3


def test_polars_loader_include_all_statuses() -> None:
    """PolarsLoader with all statuses enabled keeps all rows."""
    ds = _dataset_with_statuses(["analyzed", "awaiting", "rejected"])
    df = ds.to_polars(rejected=True)
    assert len(df) == 3


# ============================================================================
# File loaders are LOSSLESS — no status filtering
# ============================================================================


def test_parquet_loader_preserves_all_rows(tmp_path) -> None:
    """ParquetLoader preserves all rows (no status filtering)."""
    ds = _dataset_with_statuses(
        ["analyzed", "awaiting", "modified", "rejected", "deferred"]
    )
    path = tmp_path / "out.parquet"
    ds.to_parquet(path)

    reloaded = VulnDataset.from_parquet(str(path))
    assert len(reloaded) == 5


def test_json_loader_preserves_all_rows(tmp_path) -> None:
    """JSONLoader preserves all rows (no status filtering)."""
    ds = _dataset_with_statuses(
        ["analyzed", "awaiting", "modified", "rejected", "deferred"]
    )
    path = tmp_path / "out.json"
    ds.to_json(path)

    reloaded = VulnDataset.from_json(str(path))
    assert len(reloaded) == 5


def test_to_parquet_has_no_status_kwargs() -> None:
    """to_parquet() signature has no status kwargs (lossless archive)."""
    import inspect

    sig = inspect.signature(VulnDataset.to_parquet)
    param_names = set(sig.parameters.keys())
    assert "analyzed" not in param_names
    assert "rejected" not in param_names
    assert "awaiting" not in param_names


def test_to_json_has_no_status_kwargs() -> None:
    """to_json() signature has no status kwargs (lossless archive)."""
    import inspect

    sig = inspect.signature(VulnDataset.to_json)
    param_names = set(sig.parameters.keys())
    assert "analyzed" not in param_names
    assert "rejected" not in param_names
    assert "awaiting" not in param_names


# ============================================================================
# NVD fixture integration
# ============================================================================


def test_nvd_fixture_default_filter() -> None:
    """NVD sample fixture: default DataFrame filter excludes only rejected."""
    ds = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")
    # Fixture has 5 records: analyzed, awaiting, modified, analyzed(pre-2017), rejected(pre-2017)
    assert len(ds) == 5

    df = ds.to_pandas()
    # only rejected excluded by default → 4 remain
    assert len(df) == 4
    assert "rejected" not in df["nvd_status"].tolist()


def test_nvd_fixture_all_statuses() -> None:
    """NVD sample fixture: include all statuses keeps 5 records."""
    ds = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")
    df = ds.to_pandas(rejected=True)
    assert len(df) == 5
