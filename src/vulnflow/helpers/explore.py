"""Explorer - LLM-friendly functions for interactive vulnerability analysis.

All functions return JSON-serializable dicts/lists.

Public API (19): load, data_info, describe, summary, severity, trend,
nested, search, lookup, status_distribution,
describe_severity, describe_nested, describe_yearly, describe_monthly,
risk_priority, exploitation, attack_surface, vendor_risk, weakness_profile.
"""

from __future__ import annotations

import functools
import logging
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from vulnflow.utils import json as vjson

if TYPE_CHECKING:
    from vulnflow.dataset import VulnDataset

logger = logging.getLogger(__name__)

DEFAULT_CACHE_PATH = Path("data/complete_library.parquet")

__all__ = [
    "load",
    "data_info",
    "describe",
    "summary",
    "severity",
    "trend",
    "nested",
    "search",
    "lookup",
    "status_distribution",
    "describe_severity",
    "describe_nested",
    "describe_yearly",
    "describe_monthly",
    "risk_priority",
    "exploitation",
    "attack_surface",
    "vendor_risk",
    "weakness_profile",
]

_AV_ALIASES: dict[str, str] = {
    "NETWORK": "N",
    "ADJACENT": "A",
    "LOCAL": "L",
    "PHYSICAL": "P",
}
_NVD_FIRST_YEAR = 2002
_NESTED_COLUMNS = frozenset(
    {"cvss_scores", "cwes", "cpes", "references", "exploits", "epss_scores", "kev"}
)
_KNOWN_NUMERIC = frozenset(
    {
        "cvss_base_score",
        "cvss_exploitability_score",
        "cvss_impact_score",
        "epss_probability",
        "epss_percentile",
        "epss_model",
    }
)
_IDENTITY_COLS = frozenset({"id", "cve_id", "description"})


def _supports_fmt(fn: Any) -> Any:
    """Wrap function to accept fmt='json' for JSON string output."""

    @functools.wraps(fn)
    def wrapper(*args: Any, fmt: str = "dict", **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        if fmt == "json":
            import json as stdlib_json

            return stdlib_json.dumps(result, indent=2, default=str)
        return result

    return wrapper


def _analysis_mask(ds: VulnDataset) -> np.ndarray:
    """Boolean mask excluding rejected records."""
    status = ds.get_column_or_none("nvd_status")
    if status is None:
        return np.ones(len(ds), dtype=bool)
    result: np.ndarray = status != "rejected"
    return result


def _normalize_attack_vector(av: str) -> str:
    return _AV_ALIASES.get(av.upper(), av.upper())


def _valid_mask(arr: np.ndarray) -> np.ndarray:
    """Boolean mask of non-null/non-empty elements."""
    result: np.ndarray
    if arr.dtype == np.dtype(object):
        result = (arr != None) & (arr != "")  # noqa: E711
    elif arr.dtype.kind == "f":
        result = ~np.isnan(arr)
    elif arr.dtype.kind == "M":
        result = ~np.isnat(arr)
    else:
        result = np.ones(len(arr), dtype=bool)
    return result


def _value_counts(arr: np.ndarray, n: int = 0) -> dict[str, int]:
    """Top-N value counts from a pre-filtered (no nulls) array. 0 = all."""
    if len(arr) == 0:
        return {}
    unique, counts = np.unique(arr, return_counts=True)
    order = np.argsort(-counts)
    if n > 0:
        order = order[:n]
    return {str(unique[i]): int(counts[i]) for i in order}


def _extract_years(cve_ids: np.ndarray) -> np.ndarray:
    """Extract year ints from CVE ID strings. Returns -1 for unparseable."""
    u = cve_ids.astype("U20")
    result = np.full(len(u), -1, dtype=np.int32)
    for i in range(len(u)):
        s = u[i]
        if len(s) >= 8 and s[4:8].isdigit():
            result[i] = int(s[4:8])
    return result


def _find_column(ds_or_cols: Any, *names: str) -> str | None:
    """Find first matching column name."""
    cols = ds_or_cols.columns if hasattr(ds_or_cols, "columns") else ds_or_cols
    for name in names:
        if name in cols:
            return name
    return None


def _get_column_or_none(ds: VulnDataset, name: str) -> np.ndarray | None:
    """Get column from dataset, returning None if missing."""
    result: np.ndarray | None = ds.get_column_or_none(name)
    return result


def _get_float_column(
    ds: VulnDataset, name: str, mask: np.ndarray
) -> np.ndarray | None:
    """Get a column as float64, coercing object arrays. Returns None if missing."""
    col = ds.get_column_or_none(name)
    if col is None:
        return None
    filtered = col[mask]
    result: np.ndarray
    if filtered.dtype == object:
        import pandas as pd_module

        result = pd_module.to_numeric(
            pd_module.Series(filtered), errors="coerce"
        ).to_numpy()
    else:
        result = filtered.astype(np.float64)
    return result


def _parse_json_cell(cell: Any) -> Any:
    """Parse a cell that may be JSON-encoded or already parsed."""
    if cell is None:
        return None
    if isinstance(cell, str):
        if not cell or cell == "null" or cell == "[]":
            return None
        try:
            return vjson.loads(cell)
        except (vjson.JSONDecodeError, ValueError):
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


def _serialize_value(val: Any) -> Any:
    """Convert a value to JSON-serializable form."""
    if val is None:
        return None
    if isinstance(val, float) and (val != val):
        return None
    if isinstance(val, (str, int, float, bool)):
        return val
    if isinstance(val, datetime):
        return val.isoformat()
    type_name = type(val).__name__
    if "int" in type_name or "float" in type_name:
        return val.item() if hasattr(val, "item") else val
    if "Timestamp" in type_name:
        return val.isoformat()
    return str(val)


def _profile_column(
    name: str, arr: np.ndarray, top_n: int, sample_json: int
) -> dict[str, Any]:
    """Classify and profile a column in one pass.

    Defers np.unique (the dominant cost) until actually needed, and
    computes it at most once per column.
    """
    n = len(arr)
    mask = _valid_mask(arr)
    fill_pct = round(float(np.sum(mask)) / n * 100, 1) if n else 0.0
    valid = arr[mask]

    # Classify cheaply first (no np.unique for identity/nested/numeric/etc.)
    kind = _classify_kind(name, arr, valid)

    info: dict[str, Any] = {
        "name": name,
        "dtype": str(arr.dtype),
        "kind": kind,
        "fill_pct": fill_pct,
    }

    if kind == "numeric":
        if arr.dtype == object:
            import pandas as pd_module

            floats = pd_module.to_numeric(
                pd_module.Series(arr), errors="coerce"
            ).to_numpy()
        else:
            floats = arr.astype(np.float64)
        valid_f = floats[~np.isnan(floats)]
        info["unique_count"] = int(len(np.unique(valid_f))) if len(valid_f) > 0 else 0
        if len(valid_f) > 0:
            info["min"] = round(float(np.min(valid_f)), 2)
            info["max"] = round(float(np.max(valid_f)), 2)
            info["mean"] = round(float(np.mean(valid_f)), 2)
            info["p50"] = round(float(np.median(valid_f)), 2)
    elif kind == "timestamp":
        ts_valid = arr[~np.isnat(arr)] if arr.dtype.kind == "M" else arr
        if len(ts_valid) > 0:
            info["min"] = str(np.min(ts_valid))
            info["max"] = str(np.max(ts_valid))
    elif kind == "bool":
        if arr.dtype == np.dtype(bool):
            info["true_count"] = int(np.sum(arr))
            info["false_count"] = int(n - np.sum(arr))
        else:
            info["true_count"] = int(sum(1 for v in arr if v is True))
            info["false_count"] = int(sum(1 for v in arr if v is False))
    elif kind == "json_array":
        info["avg_items"] = _avg_json_array_length(arr, sample_json)
    elif kind in ("categorical", "high_cardinality"):
        # Compute np.unique once — only for columns that need it
        if len(valid) > 0:
            u, c = np.unique(valid, return_counts=True)
            info["unique_count"] = int(len(u))
            limit = top_n if kind == "categorical" else min(top_n, 5)
            order = np.argsort(-c)[:limit]
            info["top_values"] = {str(u[i]): int(c[i]) for i in order}
        else:
            info["unique_count"] = 0
            info["top_values"] = {}
    elif kind == "identity":
        # Skip expensive np.unique — use fill count as proxy
        info["unique_count"] = int(np.sum(mask))

    return info


def _classify_kind(name: str, arr: np.ndarray, valid: np.ndarray) -> str:
    """Classify column kind without np.unique (cheap classification)."""
    if name in _IDENTITY_COLS:
        return "identity"
    if name in _KNOWN_NUMERIC:
        return "numeric"
    dtype = arr.dtype
    if dtype.kind in ("f", "i", "u"):
        return "numeric"
    if dtype.kind == "M":
        return "timestamp"
    if dtype == np.dtype(bool) or name == "is_kev":
        return "bool"
    if name in _NESTED_COLUMNS and dtype is np.dtype(object):
        first = next((v for v in arr if v is not None and v != ""), None)
        if isinstance(first, str):
            if first.startswith("["):
                return "json_array"
            if first.startswith("{"):
                return "json_object"
    if dtype is np.dtype(object) and len(valid) > 0:
        # Sample-based cardinality check: take first 1000 valid values
        sample = valid[:1000] if len(valid) > 1000 else valid
        uc = len(np.unique(sample))
        return "categorical" if uc <= 25 else "high_cardinality"
    return "categorical"


def _avg_json_array_length(arr: np.ndarray, sample_size: int) -> float | None:
    """Sampled average items per JSON array cell."""
    non_null = [
        v for v in arr if v is not None and v != "" and v != "[]" and v != "null"
    ]
    if not non_null:
        return None
    rng = np.random.default_rng(42)
    indices = rng.choice(
        len(non_null), size=min(len(non_null), sample_size), replace=False
    )
    lengths = [
        len(p)
        for idx in indices
        if isinstance((p := _parse_json_cell(non_null[idx])), list)
    ]
    return round(sum(lengths) / len(lengths), 1) if lengths else None


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
# Public API: Core functions
# ============================================================================


@_supports_fmt
def data_info() -> dict[str, Any]:
    """Report what vulnerability data is available and what is missing."""
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

    import time

    from vulnflow.dataset import VulnDataset

    ds = VulnDataset.from_parquet(str(DEFAULT_CACHE_PATH))
    years_arr = _extract_years(ds.get_column("cve_id"))
    sorted_years = sorted({int(y) for y in years_arr if y > 0})
    expected = set(range(_NVD_FIRST_YEAR, current_year + 1))
    missing = sorted(expected - set(sorted_years))
    age_hours = (time.time() - DEFAULT_CACHE_PATH.stat().st_mtime) / 3600

    return {
        "has_cache": True,
        "cache_path": str(DEFAULT_CACHE_PATH),
        "record_count": len(ds),
        "years": sorted_years,
        "year_range": (
            f"{sorted_years[0]}-{sorted_years[-1]}" if sorted_years else None
        ),
        "is_complete": len(missing) == 0,
        "missing_years": missing,
        "cache_age_hours": round(age_hours, 1),
        "needs_update": age_hours > 24,
        "columns": ds.columns,
    }


def load(path: str | Path | None = None) -> VulnDataset:
    """Load dataset from parquet cache, or build if missing.

    Args:
        path: Explicit path (.parquet or .json). If None, uses default cache.
    """
    from vulnflow.dataset import VulnDataset

    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
        return _load_by_extension(p)

    if DEFAULT_CACHE_PATH.exists():
        logger.info("Loading from cache: %s", DEFAULT_CACHE_PATH)
        return VulnDataset.from_parquet(str(DEFAULT_CACHE_PATH))

    logger.info("No cache found, building via CacheBuilder...")
    from vulnflow.helpers.builder import CacheBuilder

    ds = CacheBuilder(quiet=True).build()
    DEFAULT_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(DEFAULT_CACHE_PATH)
    logger.info("Cache saved to %s", DEFAULT_CACHE_PATH)
    return ds


@_supports_fmt
def describe(
    ds: VulnDataset, *, top_n: int = 10, sample_json: int = 1000
) -> dict[str, Any]:
    """Adaptive statistical profile of all columns."""
    return {
        "record_count": len(ds),
        "column_count": len(ds.columns),
        "columns": [
            _profile_column(c, ds.get_column(c), top_n, sample_json) for c in ds.columns
        ],
    }


@_supports_fmt
def summary(ds: VulnDataset) -> dict[str, Any]:
    """High-level overview: counts, years, severity, top CWEs/vendors."""
    mask = _analysis_mask(ds)
    filtered_ids = ds.get_column("cve_id")[mask]

    result: dict[str, Any] = {
        "record_count": int(np.sum(mask)),
        "columns": ds.columns,
    }

    years = _extract_years(filtered_ids)
    valid_years = years[years > 0]
    if len(valid_years) > 0:
        uy, cy = np.unique(valid_years, return_counts=True)
        result["years"] = {int(y): int(c) for y, c in zip(uy, cy, strict=True)}

    result["status"] = status_distribution(ds)

    for key, col_name in [
        ("severity", "cvss_base_severity"),
        ("top_cwes", _find_column(ds, "primary_cwe_id", "cwe_id")),
        ("top_vendors", _find_column(ds, "primary_vendor", "cpe_vendor")),
    ]:
        if col_name is None:
            continue
        col = ds.get_column_or_none(col_name)
        if col is None:
            continue
        filtered = col[mask]
        vm = _valid_mask(filtered)
        valid = filtered[vm]
        if len(valid) > 0:
            result[key] = _value_counts(valid, 10 if key != "severity" else 0)

    return result


@_supports_fmt
def severity(
    ds: VulnDataset,
    *,
    attack_vector: str | None = None,
    min_score: float | None = None,
) -> dict[str, Any]:
    """Severity breakdown with per-year detail and optional filters."""
    mask = _analysis_mask(ds)

    if attack_vector is not None:
        av_col = ds.get_column_or_none("cvss_attack_vector")
        if av_col is not None:
            mask = mask & (av_col == _normalize_attack_vector(attack_vector))

    if min_score is not None:
        score_col = ds.get_column_or_none("cvss_base_score")
        if score_col is not None:
            import pandas as pd_module

            scores = pd_module.to_numeric(
                pd_module.Series(score_col), errors="coerce"
            ).to_numpy()
            mask = mask & (~np.isnan(scores) & (scores >= min_score))

    result: dict[str, Any] = {"total": int(np.sum(mask))}

    sev_col = ds.get_column_or_none("cvss_base_severity")
    if sev_col is None:
        result["overall"] = {}
        result["by_year"] = {}
        return result

    filtered_sev = sev_col[mask]
    sev_vm = _valid_mask(filtered_sev)
    result["overall"] = _value_counts(filtered_sev[sev_vm])

    # Per-year
    years = _extract_years(ds.get_column("cve_id")[mask])
    by_year: dict[str, dict[str, int]] = {}
    for yv in sorted(np.unique(years[years > 0])):
        ym = years == yv
        ys = filtered_sev[ym]
        ys_valid = ys[_valid_mask(ys)]
        if len(ys_valid) > 0:
            by_year[str(int(yv))] = _value_counts(ys_valid)
    result["by_year"] = by_year
    return result


@_supports_fmt
def trend(
    ds: VulnDataset, *, by: str = "year", year: int | None = None
) -> dict[str, int]:
    """Temporal CVE counts. by='year' or 'month', with optional year filter."""
    mask = _analysis_mask(ds)

    if by == "year":
        years = _extract_years(ds.get_column("cve_id")[mask])
        valid = years[years > 0]
        if len(valid) == 0:
            return {}
        unique_y, counts_y = np.unique(valid, return_counts=True)
        order = np.argsort(unique_y)
        return {str(int(unique_y[i])): int(counts_y[i]) for i in order}

    # by == "month"
    pub_col = ds.get_column_or_none("nvd_published_at")
    if pub_col is None:
        return {}

    filtered_pub = pub_col[mask]

    # Convert numpy datetime64 to Python datetime for .year/.month access
    if filtered_pub.dtype.kind == "M":
        # Vectorized: extract year and month from datetime64 array
        valid_dt = ~np.isnat(filtered_pub)
        ts = filtered_pub[valid_dt].astype("datetime64[M]")
        year_arr = ts.astype("datetime64[Y]").astype(int) + 1970
        month_arr = (ts - ts.astype("datetime64[Y]")).astype(int) + 1
        counts: Counter[str] = Counter()
        for i in range(len(year_arr)):
            y, m = int(year_arr[i]), int(month_arr[i])
            if year is not None and y != year:
                continue
            counts[f"{y:04d}-{m:02d}"] += 1
        return dict(sorted(counts.items()))

    # Fallback for object arrays with Python datetime values
    counts_obj: Counter[str] = Counter()
    for val in filtered_pub:
        if val is None:
            continue
        try:
            if isinstance(val, np.datetime64):
                if np.isnat(val):
                    continue
                dt = val.astype("datetime64[us]").item()
            elif isinstance(val, datetime):
                dt = val
            else:
                continue
            val_year = dt.year
            if year is not None and val_year != year:
                continue
            counts_obj[f"{val_year:04d}-{dt.month:02d}"] += 1
        except (AttributeError, TypeError, OverflowError):
            continue

    return dict(sorted(counts_obj.items()))


@_supports_fmt
def nested(
    ds: VulnDataset, column: str, field: str | None = None, n: int = 10
) -> dict[str, int]:
    """Top values from a nested JSON column."""
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
        return {}
    counter = Counter(str(v) for v in values if v is not None)
    return dict(counter.most_common(n))


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
    attack_vector: str | None = None,
    is_kev: bool | None = None,
    min_epss: float | None = None,
    max_epss: float | None = None,
    vendor: str | None = None,
    product: str | None = None,
    ransomware: bool | None = None,
    limit: int = 20,
    columns: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Multi-criteria CVE search returning list of dicts."""
    import pandas as pd_module

    df = ds.to_pandas(awaiting=True, deferred=True)
    mask = pd_module.Series(True, index=df.index)

    if severity is not None and "cvss_base_severity" in df.columns:
        mask = mask & (df["cvss_base_severity"].str.upper() == severity.upper())
    if min_score is not None and "cvss_base_score" in df.columns:
        mask = mask & (
            pd_module.to_numeric(df["cvss_base_score"], errors="coerce") >= min_score
        )
    if year is not None and "cve_id" in df.columns:
        mask = mask & (
            df["cve_id"].str.extract(r"CVE-(\d{4})", expand=False) == str(year)
        )
    if status is not None and "nvd_status" in df.columns:
        mask = mask & (df["nvd_status"] == status)
    if keyword is not None and "description" in df.columns:
        mask = mask & df["description"].str.contains(
            re.escape(keyword), case=False, na=False
        )
    if cwe is not None:
        cwe_col = _find_column(df, "primary_cwe_id", "cwe_id")
        if cwe_col is not None:
            mask = mask & (df[cwe_col] == cwe)
    if attack_vector is not None and "cvss_attack_vector" in df.columns:
        mask = mask & (
            df["cvss_attack_vector"] == _normalize_attack_vector(attack_vector)
        )
    if is_kev is not None and "is_kev" in df.columns:
        mask = mask & (df["is_kev"] == is_kev)
    if min_epss is not None and "epss_probability" in df.columns:
        epss_vals = pd_module.to_numeric(df["epss_probability"], errors="coerce")
        mask = mask & (epss_vals >= min_epss)
    if max_epss is not None and "epss_probability" in df.columns:
        epss_vals = pd_module.to_numeric(df["epss_probability"], errors="coerce")
        mask = mask & (epss_vals <= max_epss)
    if vendor is not None:
        vendor_col = _find_column(df, "cpe_vendor", "primary_vendor")
        if vendor_col is not None:
            mask = mask & (df[vendor_col].str.lower() == vendor.lower())
    if product is not None:
        product_col = _find_column(df, "cpe_product", "primary_product")
        if product_col is not None:
            mask = mask & (df[product_col].str.lower() == product.lower())
    if ransomware is not None and "kev_ransomware" in df.columns:
        mask = mask & (df["kev_ransomware"] == ransomware)

    if columns is None:
        preferred = [
            "cve_id",
            "description",
            "cvss_base_score",
            "cvss_base_severity",
            "nvd_status",
            "nvd_published_at",
        ]
        # Add enriched columns when present
        for extra in ["epss_probability", "is_kev"]:
            if extra in df.columns:
                preferred.append(extra)
        for col_name in [
            _find_column(ds, "primary_cwe_id", "cwe_id"),
            _find_column(ds, "primary_vendor", "cpe_vendor"),
            _find_column(ds, "primary_product", "cpe_product"),
        ]:
            if col_name:
                preferred.append(col_name)
        columns = [c for c in preferred if c in df.columns]

    available = [c for c in columns if c in df[mask].columns]
    result_df = df[mask][available].head(limit)
    return [
        {col: _serialize_value(row[col]) for col in available}
        for _, row in result_df.iterrows()
    ]


@_supports_fmt
def lookup(ds: VulnDataset, cve_id: str) -> dict[str, Any] | None:
    """Single CVE detail as dict, or None if not found."""
    col = ds.get_column("cve_id")
    indices = np.where(col == cve_id)[0]
    if len(indices) == 0:
        return None

    idx = int(indices[0])
    record: dict[str, Any] = {}
    for col_name in ds.columns:
        val = ds.get_column(col_name)[idx]
        if col_name in _NESTED_COLUMNS and isinstance(val, str):
            parsed = _parse_json_cell(val)
            record[col_name] = parsed if parsed is not None else val
        else:
            record[col_name] = _serialize_value(val)
    return record


@_supports_fmt
def status_distribution(ds: VulnDataset) -> dict[str, int]:
    """NVD status counts."""
    return ds.get_status_distribution()


# ============================================================================
# Skill aliases (delegate to core functions)
# ============================================================================


@_supports_fmt
def describe_severity(
    ds: VulnDataset,
    *,
    attack_vector: str | None = None,
    min_score: float | None = None,
) -> dict[str, Any]:
    """Alias for severity() - severity breakdown with optional filters."""
    result: dict[str, Any] = severity(
        ds, attack_vector=attack_vector, min_score=min_score
    )
    return result


@_supports_fmt
def describe_nested(
    ds: VulnDataset, column: str, field: str | None = None, n: int = 10
) -> dict[str, int]:
    """Alias for nested() - top values from a nested JSON column."""
    result: dict[str, int] = nested(ds, column, field, n)
    return result


@_supports_fmt
def describe_yearly(ds: VulnDataset) -> dict[str, int]:
    """Alias for trend(by='year') - yearly CVE counts."""
    result: dict[str, int] = trend(ds, by="year")
    return result


@_supports_fmt
def describe_monthly(ds: VulnDataset, *, year: int | None = None) -> dict[str, int]:
    """Alias for trend(by='month') - monthly CVE counts."""
    result: dict[str, int] = trend(ds, by="month", year=year)
    return result


# ============================================================================
# Domain analysis functions
# ============================================================================


@_supports_fmt
def risk_priority(ds: VulnDataset, *, limit: int = 20) -> dict[str, Any]:
    """Risk prioritization: what should I patch first?

    Combines CVSS severity + EPSS probability + KEV status into risk
    quadrants with a ranked top-N urgent list.
    """
    mask = _analysis_mask(ds)
    n = int(np.sum(mask))

    sev_col = _get_column_or_none(ds, "cvss_base_severity")
    score_col = _get_float_column(ds, "cvss_base_score", mask)
    epss_col = _get_float_column(ds, "epss_probability", mask)
    kev_col = _get_column_or_none(ds, "is_kev")

    sev_arr = sev_col[mask] if sev_col is not None else None
    kev_arr = kev_col[mask] if kev_col is not None else None

    # Build boolean masks for severity and exploitation signals
    is_high_crit = np.zeros(n, dtype=bool)
    if sev_arr is not None:
        valid_sev = _valid_mask(sev_arr)
        upper = np.where(valid_sev, np.char.upper(sev_arr.astype("U20")), "")
        is_high_crit = (upper == "HIGH") | (upper == "CRITICAL")

    has_kev = np.zeros(n, dtype=bool)
    if kev_arr is not None:
        if kev_arr.dtype == bool:
            has_kev = kev_arr
        else:
            has_kev = kev_arr == True  # noqa: E712

    has_high_epss = np.zeros(n, dtype=bool)
    epss_valid = np.zeros(n, dtype=bool)
    epss_vals = np.zeros(n, dtype=np.float64)
    if epss_col is not None:
        epss_valid = ~np.isnan(epss_col)
        epss_vals = np.where(epss_valid, epss_col, 0.0)
        has_high_epss = epss_vals > 0.7

    # Quadrants
    actively_exploited = has_kev | has_high_epss
    critical_exploited = is_high_crit & actively_exploited
    critical_unlikely = is_high_crit & ~actively_exploited
    sleeper = ~is_high_crit & (epss_vals > 0.3)
    low_risk = ~critical_exploited & ~critical_unlikely & ~sleeper

    quadrants = {
        "critical_exploited": int(np.sum(critical_exploited)),
        "critical_unlikely": int(np.sum(critical_unlikely)),
        "sleeper_threats": int(np.sum(sleeper)),
        "low_risk": int(np.sum(low_risk)),
    }

    # Risk score = cvss_score * epss * (2 if kev else 1)
    cvss_vals = np.zeros(n, dtype=np.float64)
    if score_col is not None:
        score_valid = ~np.isnan(score_col)
        cvss_vals = np.where(score_valid, score_col, 0.0)

    kev_mult = np.where(has_kev, 2.0, 1.0)
    risk_scores = cvss_vals * epss_vals * kev_mult

    # Top-N urgent
    top_indices = np.argsort(-risk_scores)[:limit]
    cve_ids = ds.get_column("cve_id")[mask]
    top_urgent: list[dict[str, Any]] = []
    for idx in top_indices:
        if risk_scores[idx] <= 0:
            break
        entry: dict[str, Any] = {
            "cve_id": str(cve_ids[idx]),
            "risk_score": round(float(risk_scores[idx]), 4),
        }
        if score_col is not None:
            entry["cvss_score"] = _serialize_value(score_col[idx])
        if epss_col is not None:
            entry["epss"] = _serialize_value(epss_col[idx])
        entry["is_kev"] = bool(has_kev[idx])
        top_urgent.append(entry)

    # Signal coverage
    signal_coverage = {
        "has_cvss": int(np.sum(cvss_vals > 0)),
        "has_epss": int(np.sum(epss_valid)),
        "has_kev": int(np.sum(has_kev)),
        "has_all_three": int(np.sum((cvss_vals > 0) & epss_valid & has_kev)),
    }

    return {
        "total": n,
        "quadrants": quadrants,
        "top_urgent": top_urgent,
        "signal_coverage": signal_coverage,
    }


@_supports_fmt
def exploitation(ds: VulnDataset) -> dict[str, Any]:
    """Exploitation intelligence: what's being exploited?

    Three sections: KEV stats, EPSS distribution, overlap analysis.
    """
    mask = _analysis_mask(ds)
    n = int(np.sum(mask))
    result: dict[str, Any] = {"total": n}

    # KEV section
    kev_col = _get_column_or_none(ds, "is_kev")
    kev_section: dict[str, Any] = {"total": 0}
    has_kev = np.zeros(n, dtype=bool)

    if kev_col is not None:
        kev_arr = kev_col[mask]
        if kev_arr.dtype == bool:
            has_kev = kev_arr
        else:
            has_kev = kev_arr == True  # noqa: E712
        kev_section["total"] = int(np.sum(has_kev))

        # Ransomware count
        ransomware_col = _get_column_or_none(ds, "kev_ransomware")
        if ransomware_col is not None:
            rw_arr = ransomware_col[mask]
            if rw_arr.dtype == bool:
                kev_section["ransomware"] = int(np.sum(rw_arr & has_kev))
            else:
                rw_mask = (rw_arr == True) | (rw_arr == "Known")  # noqa: E712
                kev_section["ransomware"] = int(np.sum(rw_mask & has_kev))

        # Top vendors
        vendor_col = _get_column_or_none(ds, "kev_vendor")
        if vendor_col is not None:
            kev_vendors = vendor_col[mask][has_kev]
            vm = _valid_mask(kev_vendors)
            if np.sum(vm) > 0:
                kev_section["top_vendors"] = _value_counts(kev_vendors[vm], 10)

        # Recent additions (30d, 90d) and overdue
        _add_kev_date_stats(ds, mask, has_kev, kev_section)

    result["kev"] = kev_section

    # EPSS section
    epss_col = _get_float_column(ds, "epss_probability", mask)
    epss_section: dict[str, Any] = {}
    has_high_epss = np.zeros(n, dtype=bool)

    if epss_col is not None:
        epss_valid = ~np.isnan(epss_col)
        valid_epss = epss_col[epss_valid]
        has_high_epss = epss_valid & (epss_col > 0.5)

        epss_section["high"] = int(np.sum(epss_col[epss_valid] > 0.5))
        epss_section["very_high"] = int(np.sum(epss_col[epss_valid] > 0.9))
        if len(valid_epss) > 0:
            epss_section["percentiles"] = {
                "p50": round(float(np.percentile(valid_epss, 50)), 4),
                "p90": round(float(np.percentile(valid_epss, 90)), 4),
                "p99": round(float(np.percentile(valid_epss, 99)), 4),
            }

    result["epss"] = epss_section

    # Overlap section
    overlap: dict[str, int] = {}
    overlap["kev_and_high_epss"] = int(np.sum(has_kev & has_high_epss))
    if epss_col is not None:
        epss_valid_mask = ~np.isnan(epss_col)
        low_epss = epss_valid_mask & (epss_col < 0.1)
        overlap["kev_and_low_epss"] = int(np.sum(has_kev & low_epss))
        overlap["high_epss_not_kev"] = int(np.sum(has_high_epss & ~has_kev))

    result["overlap"] = overlap
    return result


def _add_kev_date_stats(
    ds: VulnDataset,
    mask: np.ndarray,
    has_kev: np.ndarray,
    section: dict[str, Any],
) -> None:
    """Add date-based KEV stats (recent additions, overdue)."""
    from datetime import UTC, timedelta

    now = datetime.now(UTC)

    date_added_col = _get_column_or_none(ds, "kev_date_added")
    if date_added_col is not None:
        added_arr = date_added_col[mask][has_kev]
        recent_30d = 0
        recent_90d = 0
        cutoff_30 = now - timedelta(days=30)
        cutoff_90 = now - timedelta(days=90)

        for val in added_arr:
            if val is None:
                continue
            try:
                if isinstance(val, str):
                    dt = datetime.fromisoformat(val)
                elif isinstance(val, np.datetime64):
                    if np.isnat(val):
                        continue
                    dt = val.astype("datetime64[us]").item()
                else:
                    dt = val
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=UTC)
                if dt >= cutoff_30:
                    recent_30d += 1
                if dt >= cutoff_90:
                    recent_90d += 1
            except (AttributeError, TypeError, ValueError, OverflowError):
                continue

        section["recent_30d"] = recent_30d
        section["recent_90d"] = recent_90d

    due_col = _get_column_or_none(ds, "kev_due_date")
    if due_col is not None:
        due_arr = due_col[mask][has_kev]
        overdue = 0
        for val in due_arr:
            if val is None:
                continue
            try:
                if isinstance(val, str):
                    dt = datetime.fromisoformat(val)
                elif isinstance(val, np.datetime64):
                    if np.isnat(val):
                        continue
                    dt = val.astype("datetime64[us]").item()
                else:
                    dt = val
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=UTC)
                if dt < now:
                    overdue += 1
            except (AttributeError, TypeError, ValueError, OverflowError):
                continue
        section["overdue"] = overdue


@_supports_fmt
def attack_surface(ds: VulnDataset) -> dict[str, Any]:
    """Attack surface profile: CVSS component distributions and worst-case metrics."""
    mask = _analysis_mask(ds)
    result: dict[str, Any] = {"total": int(np.sum(mask))}

    # Component distributions
    component_map = {
        "by_vector": "cvss_attack_vector",
        "by_complexity": "cvss_attack_complexity",
        "by_privileges": "cvss_privileges_required",
        "by_interaction": "cvss_user_interaction",
    }
    for key, col_name in component_map.items():
        col = _get_column_or_none(ds, col_name)
        if col is not None:
            filtered = col[mask]
            vm = _valid_mask(filtered)
            result[key] = _value_counts(filtered[vm])
        else:
            result[key] = {}

    # Worst-case metrics
    av_col = _get_column_or_none(ds, "cvss_attack_vector")
    pr_col = _get_column_or_none(ds, "cvss_privileges_required")
    ui_col = _get_column_or_none(ds, "cvss_user_interaction")
    scope_col = _get_column_or_none(ds, "cvss_scope")

    worst_case: dict[str, int] = {}
    if av_col is not None and pr_col is not None:
        av = av_col[mask]
        pr = pr_col[mask]
        remote_no_auth = (av == "N") & ((pr == "N") | (pr == "NONE"))
        worst_case["no_auth_remote"] = int(np.sum(remote_no_auth))

        if ui_col is not None:
            ui = ui_col[mask]
            no_interaction = remote_no_auth & ((ui == "N") | (ui == "NONE"))
            worst_case["no_interaction"] = int(np.sum(no_interaction))

    if scope_col is not None:
        sc = scope_col[mask]
        worst_case["scope_changed"] = int(np.sum((sc == "C") | (sc == "CHANGED")))

    result["worst_case"] = worst_case

    # CIA impact distributions
    cia: dict[str, dict[str, int]] = {}
    for impact_name, col_name in [
        ("confidentiality", "cvss_confidentiality"),
        ("integrity", "cvss_integrity"),
        ("availability", "cvss_availability"),
    ]:
        col = _get_column_or_none(ds, col_name)
        if col is not None:
            filtered = col[mask]
            vm = _valid_mask(filtered)
            cia[impact_name] = _value_counts(filtered[vm])

    result["by_cia"] = cia
    return result


@_supports_fmt
def vendor_risk(
    ds: VulnDataset, vendor: str | None = None, *, top_n: int = 15
) -> dict[str, Any]:
    """Vendor risk analysis: top vendors by risk score or single vendor detail."""
    mask = _analysis_mask(ds)

    vendor_col_name = _find_column(ds, "cpe_vendor", "primary_vendor")
    if vendor_col_name is None:
        return {"error": "No vendor column found (run with cpe=True)"}

    vendor_arr = ds.get_column(vendor_col_name)[mask]
    vm = _valid_mask(vendor_arr)

    if vendor is not None:
        return _vendor_detail(ds, mask, vendor_arr, vm, vendor)

    # Top-N vendors by composite risk score
    unique_vendors, vendor_counts = np.unique(vendor_arr[vm], return_counts=True)
    if len(unique_vendors) == 0:
        return {"vendors": []}

    score_col = _get_float_column(ds, "cvss_base_score", mask)
    kev_col = _get_column_or_none(ds, "is_kev")
    kev_arr = kev_col[mask] if kev_col is not None else None

    vendors_list: list[dict[str, Any]] = []
    for v, count in zip(unique_vendors, vendor_counts, strict=True):
        v_mask = vm & (vendor_arr == v)
        entry: dict[str, Any] = {"vendor": str(v), "count": int(count)}

        avg_cvss = 0.0
        if score_col is not None:
            v_scores = score_col[v_mask]
            valid_scores = v_scores[~np.isnan(v_scores)]
            if len(valid_scores) > 0:
                avg_cvss = float(np.mean(valid_scores))
                entry["avg_cvss"] = round(avg_cvss, 2)

        kev_ratio = 0.0
        if kev_arr is not None:
            v_kev = kev_arr[v_mask]
            if v_kev.dtype == bool:
                kev_count = int(np.sum(v_kev))
            else:
                kev_count = int(np.sum(v_kev == True))  # noqa: E712
            entry["kev_count"] = kev_count
            if count > 0:
                kev_ratio = kev_count / count

        # Composite: count * avg_cvss * (1 + kev_ratio)
        entry["risk_score"] = round(count * avg_cvss * (1 + kev_ratio), 2)
        vendors_list.append(entry)

    vendors_list.sort(key=lambda x: x["risk_score"], reverse=True)
    return {"vendors": vendors_list[:top_n]}


def _vendor_detail(
    ds: VulnDataset,
    mask: np.ndarray,
    vendor_arr: np.ndarray,
    vm: np.ndarray,
    vendor: str,
) -> dict[str, Any]:
    """Single-vendor detail profile."""
    v_mask = vm & (np.char.lower(vendor_arr.astype("U100")) == vendor.lower())
    count = int(np.sum(v_mask))
    if count == 0:
        return {"vendor": vendor, "count": 0}

    result: dict[str, Any] = {"vendor": vendor, "count": count}

    # Severity distribution
    sev_col = _get_column_or_none(ds, "cvss_base_severity")
    if sev_col is not None:
        sev_arr = sev_col[mask][v_mask]
        sev_vm = _valid_mask(sev_arr)
        result["severity"] = _value_counts(sev_arr[sev_vm])

    # EPSS distribution
    epss_col = _get_float_column(ds, "epss_probability", mask)
    if epss_col is not None:
        v_epss = epss_col[v_mask]
        valid_epss = v_epss[~np.isnan(v_epss)]
        if len(valid_epss) > 0:
            result["epss"] = {
                "mean": round(float(np.mean(valid_epss)), 4),
                "p90": round(float(np.percentile(valid_epss, 90)), 4),
            }

    # KEV count
    kev_col = _get_column_or_none(ds, "is_kev")
    if kev_col is not None:
        v_kev = kev_col[mask][v_mask]
        if v_kev.dtype == bool:
            result["kev_count"] = int(np.sum(v_kev))
        else:
            result["kev_count"] = int(np.sum(v_kev == True))  # noqa: E712

    # Top products
    product_col_name = _find_column(ds, "cpe_product", "primary_product")
    if product_col_name is not None:
        prod_arr = ds.get_column(product_col_name)[mask][v_mask]
        prod_vm = _valid_mask(prod_arr)
        if np.sum(prod_vm) > 0:
            result["top_products"] = _value_counts(prod_arr[prod_vm], 10)

    # Top CWEs
    cwe_col_name = _find_column(ds, "cwe_id", "primary_cwe_id")
    if cwe_col_name is not None:
        cwe_arr = ds.get_column(cwe_col_name)[mask][v_mask]
        cwe_vm = _valid_mask(cwe_arr)
        if np.sum(cwe_vm) > 0:
            result["top_cwes"] = _value_counts(cwe_arr[cwe_vm], 10)

    return result


@_supports_fmt
def weakness_profile(
    ds: VulnDataset, *, year: int | None = None, top_n: int = 15
) -> dict[str, Any]:
    """Weakness profile: top CWEs with enriched metrics per weakness."""
    mask = _analysis_mask(ds)

    cwe_col_name = _find_column(ds, "cwe_id", "primary_cwe_id")
    if cwe_col_name is None:
        return {"error": "No CWE column found (run with cwe=True)"}

    cwe_arr = ds.get_column(cwe_col_name)[mask]
    cwe_vm = _valid_mask(cwe_arr)

    # Optional year filter
    if year is not None:
        cve_ids = ds.get_column("cve_id")[mask]
        years = _extract_years(cve_ids)
        year_mask = years == year
        cwe_vm = cwe_vm & year_mask

    valid_cwes = cwe_arr[cwe_vm]
    if len(valid_cwes) == 0:
        return {"weaknesses": []}

    unique_cwes, cwe_counts = np.unique(valid_cwes, return_counts=True)
    order = np.argsort(-cwe_counts)[:top_n]

    score_col = _get_float_column(ds, "cvss_base_score", mask)
    epss_col = _get_float_column(ds, "epss_probability", mask)
    kev_col = _get_column_or_none(ds, "is_kev")
    kev_arr = kev_col[mask] if kev_col is not None else None

    cwe_name_col_name = _find_column(ds, "cwe_name", "primary_cwe_name")
    cwe_name_arr = ds.get_column(cwe_name_col_name)[mask] if cwe_name_col_name else None

    weaknesses: list[dict[str, Any]] = []
    for idx in order:
        cwe_id = str(unique_cwes[idx])
        cwe_mask = cwe_vm & (cwe_arr == cwe_id)
        entry: dict[str, Any] = {
            "id": cwe_id,
            "count": int(cwe_counts[idx]),
        }

        # CWE name
        if cwe_name_arr is not None:
            names = cwe_name_arr[cwe_mask]
            name_vm = _valid_mask(names)
            valid_names = names[name_vm]
            if len(valid_names) > 0:
                entry["name"] = str(valid_names[0])

        # Average CVSS
        if score_col is not None:
            c_scores = score_col[cwe_mask]
            valid_scores = c_scores[~np.isnan(c_scores)]
            if len(valid_scores) > 0:
                entry["avg_cvss"] = round(float(np.mean(valid_scores)), 2)

        # Average EPSS
        if epss_col is not None:
            c_epss = epss_col[cwe_mask]
            valid_epss = c_epss[~np.isnan(c_epss)]
            if len(valid_epss) > 0:
                entry["avg_epss"] = round(float(np.mean(valid_epss)), 4)

        # KEV count
        if kev_arr is not None:
            c_kev = kev_arr[cwe_mask]
            if c_kev.dtype == bool:
                entry["kev_count"] = int(np.sum(c_kev))
            else:
                entry["kev_count"] = int(np.sum(c_kev == True))  # noqa: E712

        weaknesses.append(entry)

    result: dict[str, Any] = {"weaknesses": weaknesses}

    # Year comparison if year filter provided
    if year is not None:
        prior_year = year - 1
        cve_ids_all = ds.get_column("cve_id")[mask]
        years_all = _extract_years(cve_ids_all)
        prior_mask = cwe_vm & (years_all == prior_year)  # recompute without year filter
        # Need mask without year filter for prior year
        cwe_vm_all = _valid_mask(cwe_arr)
        prior_mask = cwe_vm_all & (years_all == prior_year)

        if np.sum(prior_mask) > 0:
            prior_cwes = cwe_arr[prior_mask]
            prior_unique, prior_counts = np.unique(prior_cwes, return_counts=True)
            prior_map = {
                str(c): int(n) for c, n in zip(prior_unique, prior_counts, strict=True)
            }

            rising: list[str] = []
            declining: list[str] = []
            for w in weaknesses:
                wid = w["id"]
                current = w["count"]
                prior = prior_map.get(wid, 0)
                if prior > 0 and current > prior * 1.2:
                    rising.append(wid)
                elif prior > 0 and current < prior * 0.8:
                    declining.append(wid)
            result["rising"] = rising
            result["declining"] = declining

    return result
