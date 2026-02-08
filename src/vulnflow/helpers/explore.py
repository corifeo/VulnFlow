"""
Explorer - Convenience functions for interactive vulnerability data analysis

Provides two layers:
1. LLM-friendly functions (return dicts/lists, JSON-serializable):
   - data_info: Report what data is available and coverage gaps
   - summary: Full dataset overview as structured dict
   - describe_severity: Severity breakdown as dict
   - describe_yearly: Year-over-year counts as dict
   - describe_monthly: Monthly counts as dict
   - describe_nested: Top values from nested JSON columns as dict
   - search: Filter CVEs and return list of dicts
   - lookup: Single CVE detail as dict

2. DataFrame functions (return pandas types, for notebooks):
   - severity_distribution, top_values, yearly_counts, monthly_counts
   - inspect_nested, find

Both layers share: load, status_distribution

Layer: User-facing helpers
Dependencies: dataset, helpers.builder
Used by: Notebooks, user scripts, Claude Code skill
"""

from __future__ import annotations

import functools
import json
import json as _json_mod
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

    from vulnflow.dataset import VulnDataset

logger = logging.getLogger(__name__)

DEFAULT_CACHE_PATH = Path("data/nvd_library.parquet")


def _supports_fmt(fn: Any) -> Any:
    """Add fmt parameter to explore functions.

    When fmt="json", returns formatted JSON string instead of dict/list.
    Default fmt="dict" preserves original return type.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, fmt: str = "dict", **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        if fmt == "json":
            return _json_mod.dumps(result, indent=2, default=str)
        return result

    return wrapper

# NVD feed years range
_NVD_FIRST_YEAR = 2002


# ============================================================================
# Data Readiness
# ============================================================================


@_supports_fmt
def data_info() -> dict[str, Any]:
    """Report what vulnerability data is available and what is missing.

    Returns a dict describing cache status, year coverage, record count,
    and freshness. Use this to decide whether to build or update before
    answering a question.

    Returns:
        Dict with keys: has_cache, cache_path, record_count, years,
        year_range, is_complete, missing_years, cache_age_hours,
        needs_update, columns.
    """
    from datetime import UTC

    current_year = datetime.now(UTC).year

    if not DEFAULT_CACHE_PATH.exists():
        return {
            "has_cache": False,
            "cache_path": str(DEFAULT_CACHE_PATH),
            "record_count": 0,
            "years": [],
            "year_range": None,
            "is_complete": False,
            "missing_years": list(range(_NVD_FIRST_YEAR, current_year + 1)),
            "cache_age_hours": None,
            "needs_update": True,
            "columns": [],
        }

    from vulnflow.dataset import VulnDataset

    ds = VulnDataset.from_parquet(str(DEFAULT_CACHE_PATH))
    col = ds.get_column("cve_id")

    years_found: set[int] = set()
    for cve_id in col:
        if cve_id and isinstance(cve_id, str) and cve_id.startswith("CVE-"):
            parts = cve_id.split("-")
            if len(parts) >= 2 and parts[1].isdigit():
                years_found.add(int(parts[1]))

    sorted_years = sorted(years_found)
    expected_years = set(range(_NVD_FIRST_YEAR, current_year + 1))
    missing = sorted(expected_years - years_found)
    is_complete = len(missing) == 0

    import time

    mtime = DEFAULT_CACHE_PATH.stat().st_mtime
    age_hours = (time.time() - mtime) / 3600

    return {
        "has_cache": True,
        "cache_path": str(DEFAULT_CACHE_PATH),
        "record_count": len(ds),
        "years": sorted_years,
        "year_range": f"{sorted_years[0]}-{sorted_years[-1]}" if sorted_years else None,
        "is_complete": is_complete,
        "missing_years": missing,
        "cache_age_hours": round(age_hours, 1),
        "needs_update": age_hours > 24,
        "columns": ds.columns,
    }


def load(path: str | Path | None = None) -> VulnDataset:
    """Load dataset from parquet cache, or build if missing.

    Resolution order:
    1. Explicit path (parquet/json/nvd based on extension)
    2. data/nvd_library.parquet (default cache)
    3. Build via CacheBuilder (saves to default cache)

    Args:
        path: Explicit path to load from. Supports .parquet, .json extensions.
              If None, uses default cache path.

    Returns:
        VulnDataset loaded from the resolved source.

    Raises:
        FileNotFoundError: If explicit path does not exist.
        ValueError: If file extension is not supported.
    """
    from vulnflow.dataset import VulnDataset

    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
        return _load_by_extension(p)

    # Try default cache
    if DEFAULT_CACHE_PATH.exists():
        logger.info("Loading from cache: %s", DEFAULT_CACHE_PATH)
        return VulnDataset.from_parquet(str(DEFAULT_CACHE_PATH))

    # Build via CacheBuilder
    logger.info("No cache found, building via CacheBuilder...")
    from vulnflow.helpers.builder import CacheBuilder

    ds = CacheBuilder(quiet=True).build()
    DEFAULT_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(DEFAULT_CACHE_PATH)
    logger.info("Cache saved to %s", DEFAULT_CACHE_PATH)
    return ds


def _load_by_extension(path: Path) -> VulnDataset:
    """Load dataset based on file extension."""
    from vulnflow.dataset import VulnDataset

    ext = path.suffix.lower()
    if ext == ".parquet":
        return VulnDataset.from_parquet(str(path))
    if ext == ".json":
        content = path.read_text(encoding="utf-8")[:200]
        if "CVE_data_type" in content or '"vulnerabilities"' in content:
            return VulnDataset.from_nvd(path)
        return VulnDataset.from_json(path)
    msg = f"Unsupported file extension: {ext} (expected .parquet or .json)"
    raise ValueError(msg)


# ============================================================================
# LLM-Friendly Functions (return dicts/lists, JSON-serializable)
# ============================================================================


@_supports_fmt
def summary(ds: VulnDataset) -> dict[str, Any]:
    """Full dataset overview as a structured dict.

    Returns a comprehensive snapshot: record count, year range, status
    breakdown, severity breakdown, and top CWEs/vendors.

    Args:
        ds: VulnDataset to summarize.

    Returns:
        Dict with keys: record_count, years, status, severity,
        top_cwes, top_vendors, columns.
    """
    df = ds.to_pandas(awaiting=True, deferred=True)

    result: dict[str, Any] = {
        "record_count": len(df),
        "columns": list(df.columns),
    }

    # Year range
    if "cve_id" in df.columns:
        import pandas as pd_module

        year_series = df["cve_id"].str.extract(r"CVE-(\d{4})", expand=False)
        year_nums = pd_module.to_numeric(year_series, errors="coerce").dropna().astype(int)
        year_counts = year_nums.value_counts().sort_index()
        result["years"] = {int(k): int(v) for k, v in year_counts.items()}  # type: ignore[call-overload]

    # Status
    result["status"] = status_distribution(ds)

    # Severity
    if "cvss_base_severity" in df.columns:
        sev = df["cvss_base_severity"].value_counts()
        result["severity"] = {str(k): int(v) for k, v in sev.items()}

    # Top CWEs
    cwe_col = _find_cwe_column(df)
    if cwe_col is not None:
        cwe_counts = df[cwe_col].value_counts().head(10)
        result["top_cwes"] = {str(k): int(v) for k, v in cwe_counts.items()}

    # Top vendors
    vendor_col = _find_vendor_column(df)
    if vendor_col is not None:
        vendor_counts = df[vendor_col].value_counts().head(10)
        result["top_vendors"] = {str(k): int(v) for k, v in vendor_counts.items()}

    return result


@_supports_fmt
def describe_severity(ds: VulnDataset) -> dict[str, Any]:
    """Severity breakdown as a dict, with per-year detail.

    Args:
        ds: VulnDataset to analyze.

    Returns:
        Dict with keys: total, overall (severity counts), by_year.
    """
    import pandas as pd_module

    df = ds.to_pandas(awaiting=True, deferred=True)
    result: dict[str, Any] = {"total": len(df)}

    if "cvss_base_severity" not in df.columns:
        result["overall"] = {}
        result["by_year"] = {}
        return result

    sev = df["cvss_base_severity"].value_counts()
    result["overall"] = {str(k): int(v) for k, v in sev.items()}

    # Per-year
    if "cve_id" in df.columns:
        df = df.copy()
        df["_year"] = pd_module.to_numeric(
            df["cve_id"].str.extract(r"CVE-(\d{4})", expand=False),
            errors="coerce",
        )
        by_year: dict[str, dict[str, int]] = {}
        for year_val in sorted(df["_year"].dropna().unique()):
            year_df = df[df["_year"] == year_val]
            year_sev = year_df["cvss_base_severity"].value_counts()
            by_year[str(int(year_val))] = {str(k): int(v) for k, v in year_sev.items()}
        result["by_year"] = by_year

    return result


@_supports_fmt
def describe_yearly(ds: VulnDataset) -> dict[str, int]:
    """CVE counts per year as a dict.

    Args:
        ds: VulnDataset to analyze.

    Returns:
        Dict mapping year (str) to count.
    """
    df = ds.to_pandas(awaiting=True, deferred=True)
    counts = yearly_counts(df)
    return {str(int(k)): int(v) for k, v in counts.items()}  # type: ignore[call-overload]


@_supports_fmt
def describe_monthly(
    ds: VulnDataset, year: int | None = None
) -> dict[str, int]:
    """CVE counts per month as a dict.

    Args:
        ds: VulnDataset to analyze.
        year: Optional year filter.

    Returns:
        Dict mapping year-month (str) to count.
    """
    df = ds.to_pandas(awaiting=True, deferred=True)
    counts = monthly_counts(df, year=year)
    return {str(k): int(v) for k, v in counts.items()}


@_supports_fmt
def describe_nested(
    ds: VulnDataset,
    column: str,
    field: str | None = None,
    n: int = 10,
) -> dict[str, int]:
    """Top values from a nested JSON column as a dict.

    Args:
        ds: VulnDataset with the nested JSON column.
        column: Name of the JSON-encoded column.
        field: Field to extract from each nested object.
        n: Number of top values.

    Returns:
        Dict mapping value to count.
    """
    series = inspect_nested(ds, column, field, n)
    return {str(k): int(v) for k, v in series.items()}


@_supports_fmt
def search(
    ds: VulnDataset,
    *,
    severity: str | None = None,
    min_score: float | None = None,
    year: int | None = None,
    status: str | None = None,
    keyword: str | None = None,
    cwe: str | None = None,
    limit: int = 20,
    columns: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Filter CVEs and return results as a list of dicts.

    Args:
        ds: VulnDataset to search.
        severity: CVSS severity level (e.g., "CRITICAL").
        min_score: Minimum CVSS base score.
        year: CVE year.
        status: NVD status.
        keyword: Case-insensitive keyword in description.
        cwe: CWE ID (e.g., "CWE-79").
        limit: Max results to return.
        columns: Columns to include in output. Defaults to a useful subset.

    Returns:
        List of dicts, each representing a CVE record.
    """
    df = ds.to_pandas(awaiting=True, deferred=True)
    filtered = find(
        df,
        severity=severity,
        min_score=min_score,
        year=year,
        status=status,
        keyword=keyword,
        cwe=cwe,
    )

    if columns is None:
        columns = _default_output_columns(df)

    available = [c for c in columns if c in filtered.columns]
    result_df = filtered[available].head(limit)

    records: list[dict[str, Any]] = []
    for _, row in result_df.iterrows():
        record = {}
        for col in available:
            val = row[col]
            record[col] = _serialize_value(val)
        records.append(record)

    return records


@_supports_fmt
def lookup(ds: VulnDataset, cve_id: str) -> dict[str, Any] | None:
    """Look up a single CVE by ID and return full details as a dict.

    Args:
        ds: VulnDataset to search.
        cve_id: CVE identifier (e.g., "CVE-2024-1234").

    Returns:
        Dict with all available fields, or None if not found.
    """
    import numpy as np

    col = ds.get_column("cve_id")
    indices = np.where(col == cve_id)[0]
    if len(indices) == 0:
        return None

    idx = int(indices[0])
    record: dict[str, Any] = {}
    for col_name in ds.columns:
        val = ds.get_column(col_name)[idx]
        # Parse nested JSON for richer output
        if col_name in _NESTED_COLUMNS and isinstance(val, str):
            parsed = _parse_json_cell(val)
            record[col_name] = parsed if parsed is not None else val
        else:
            record[col_name] = _serialize_value(val)

    return record


_NESTED_COLUMNS = frozenset({
    "cvss_scores", "cwes", "cpes", "references",
    "exploits", "epss_scores", "kev",
})


def _default_output_columns(df: pd.DataFrame) -> list[str]:
    """Select a useful default column subset for search results."""
    preferred = [
        "cve_id", "description", "cvss_base_score", "cvss_base_severity",
        "nvd_status", "nvd_published_at",
    ]
    # Add CWE/vendor columns if they exist under any naming convention
    cwe_col = _find_cwe_column(df)
    if cwe_col:
        preferred.append(cwe_col)
    vendor_col = _find_vendor_column(df)
    if vendor_col:
        preferred.append(vendor_col)
    product_col = _find_product_column(df)
    if product_col:
        preferred.append(product_col)

    return [c for c in preferred if c in df.columns]


def _find_cwe_column(df: pd.DataFrame) -> str | None:
    """Find the CWE ID column regardless of naming convention."""
    for name in ("primary_cwe_id", "cwe_id"):
        if name in df.columns:
            return name
    return None


def _find_vendor_column(df: pd.DataFrame) -> str | None:
    """Find the vendor column regardless of naming convention."""
    for name in ("primary_vendor", "cpe_vendor"):
        if name in df.columns:
            return name
    return None


def _find_product_column(df: pd.DataFrame) -> str | None:
    """Find the product column regardless of naming convention."""
    for name in ("primary_product", "cpe_product"):
        if name in df.columns:
            return name
    return None


def _serialize_value(val: Any) -> Any:
    """Convert a value to JSON-serializable form."""
    if val is None:
        return None
    if isinstance(val, float) and (val != val):  # NaN check
        return None
    if isinstance(val, (str, int, float, bool)):
        return val
    if isinstance(val, datetime):
        return val.isoformat()
    # numpy/pandas types
    type_name = type(val).__name__
    if "int" in type_name or "float" in type_name:
        return val.item() if hasattr(val, "item") else val
    if "Timestamp" in type_name:
        return val.isoformat()
    return str(val)


# ============================================================================
# DataFrame Functions (return pandas types, for notebooks)
# ============================================================================


@_supports_fmt
def status_distribution(ds: VulnDataset) -> dict[str, int]:
    """NVD status counts.

    Args:
        ds: VulnDataset with nvd_status column.

    Returns:
        Dict mapping status values to their counts.
    """
    return ds.get_status_distribution()


def severity_distribution(df: pd.DataFrame) -> pd.Series:
    """CVSS severity value counts.

    Args:
        df: DataFrame with cvss_base_severity column.

    Returns:
        Series with severity values as index and counts as values.

    Raises:
        KeyError: If cvss_base_severity column is missing.
    """
    if "cvss_base_severity" not in df.columns:
        msg = "Column 'cvss_base_severity' not found. Use to_pandas() on a dataset with CVSS flatten applied."
        raise KeyError(msg)
    return df["cvss_base_severity"].value_counts()


def top_values(df: pd.DataFrame, column: str, n: int = 10) -> pd.Series:
    """Top N most frequent values for any column.

    Args:
        df: DataFrame to analyze.
        column: Column name to count values from.
        n: Number of top values to return.

    Returns:
        Series with values as index and counts as values, top N.

    Raises:
        KeyError: If column is not found.
    """
    if column not in df.columns:
        msg = f"Column '{column}' not found. Available: {list(df.columns)}"
        raise KeyError(msg)
    return df[column].value_counts().head(n)


def yearly_counts(df: pd.DataFrame) -> pd.Series:
    """CVE counts per year, extracted from cve_id.

    Args:
        df: DataFrame with cve_id column.

    Returns:
        Series with years as index and counts as values, sorted by year.

    Raises:
        KeyError: If cve_id column is missing.
    """
    import pandas as pd_module

    if "cve_id" not in df.columns:
        msg = "Column 'cve_id' not found."
        raise KeyError(msg)
    years = df["cve_id"].str.extract(r"CVE-(\d{4})", expand=False)
    years = pd_module.to_numeric(years, errors="coerce").dropna().astype(int)
    return years.value_counts().sort_index()


def monthly_counts(df: pd.DataFrame, year: int | None = None) -> pd.Series:
    """CVE counts per month from nvd_published_at.

    Args:
        df: DataFrame with nvd_published_at column.
        year: Optional year filter.

    Returns:
        Series with year-month as index, counts as values.

    Raises:
        KeyError: If nvd_published_at column is missing.
    """
    import pandas as pd_module

    if "nvd_published_at" not in df.columns:
        msg = "Column 'nvd_published_at' not found."
        raise KeyError(msg)

    dates = pd_module.to_datetime(df["nvd_published_at"], errors="coerce")
    if year is not None:
        dates = dates[dates.dt.year == year]

    months = dates.dropna().dt.to_period("M")
    return months.value_counts().sort_index()


def inspect_nested(
    ds: VulnDataset,
    column: str,
    field: str | None = None,
    n: int = 10,
) -> pd.Series:
    """Parse a JSON column, extract values, return top N.

    Args:
        ds: VulnDataset with the nested JSON column.
        column: Name of the JSON-encoded column.
        field: Optional field name to extract from each nested object.
        n: Number of top values to return.

    Returns:
        Series with extracted values as index and counts as values, top N.

    Raises:
        KeyError: If column is not found in dataset.
    """
    import pandas as pd_module

    col_data = ds.get_column(column)

    values: list[Any] = []
    for cell in col_data:
        parsed = _parse_json_cell(cell)
        if parsed is None:
            continue
        if isinstance(parsed, list):
            for item in parsed:
                values.append(_extract_field(item, field))
        else:
            values.append(_extract_field(parsed, field))

    if not values:
        return pd_module.Series(dtype=object)

    return pd_module.Series(values).value_counts().head(n)


def find(
    df: pd.DataFrame,
    *,
    severity: str | None = None,
    min_score: float | None = None,
    year: int | None = None,
    status: str | None = None,
    keyword: str | None = None,
    cwe: str | None = None,
) -> pd.DataFrame:
    """Filter CVEs by common criteria. All filters are AND-combined.

    Args:
        df: DataFrame to filter.
        severity: CVSS severity level (e.g., "CRITICAL", "HIGH").
        min_score: Minimum CVSS base score.
        year: CVE year (extracted from cve_id).
        status: NVD status (e.g., "analyzed", "modified").
        keyword: Case-insensitive keyword search in description.
        cwe: CWE ID to match (e.g., "CWE-79").

    Returns:
        Filtered DataFrame matching all criteria.
    """
    import pandas as pd_module

    mask = pd_module.Series(True, index=df.index)

    if severity is not None and "cvss_base_severity" in df.columns:
        mask = mask & (df["cvss_base_severity"].str.upper() == severity.upper())

    if min_score is not None and "cvss_base_score" in df.columns:
        scores = pd_module.to_numeric(df["cvss_base_score"], errors="coerce")
        mask = mask & (scores >= min_score)

    if year is not None and "cve_id" in df.columns:
        cve_years = df["cve_id"].str.extract(r"CVE-(\d{4})", expand=False)
        mask = mask & (cve_years == str(year))

    if status is not None and "nvd_status" in df.columns:
        mask = mask & (df["nvd_status"] == status)

    if keyword is not None and "description" in df.columns:
        mask = mask & df["description"].str.contains(
            re.escape(keyword), case=False, na=False
        )

    if cwe is not None:
        cwe_col = _find_cwe_column(df)
        if cwe_col is not None:
            mask = mask & (df[cwe_col] == cwe)

    return df[mask]


# ============================================================================
# Internal Helpers
# ============================================================================


def _parse_json_cell(cell: Any) -> Any:
    """Parse a single cell value that may be JSON-encoded or already parsed."""
    if cell is None:
        return None
    if isinstance(cell, str):
        if not cell or cell == "null" or cell == "[]":
            return None
        try:
            return json.loads(cell)
        except (json.JSONDecodeError, ValueError):
            return None
    if isinstance(cell, (dict, list)):
        return cell if cell else None
    return None


def _extract_field(item: Any, field: str | None) -> Any:
    """Extract a field from an item (dict or object)."""
    if field is None:
        return str(item) if not isinstance(item, str) else item
    if isinstance(item, dict):
        return item.get(field)
    return getattr(item, field, None)
