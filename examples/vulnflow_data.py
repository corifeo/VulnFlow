"""
VulnFlow Data Management Library

Support functions for downloading and managing NVD vulnerability data in notebooks.

Usage:
    from vulnflow_data import download_modified, download_year_range, build_cache
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Literal
from urllib.request import urlopen, urlretrieve

from vulnflow import VulnDataset
from vulnflow.transforms.computed import CVDStateTransform
from vulnflow.transforms.flatten import (
    CPEPrimaryTransform,
    CVSSFlattenTransform,
    CWEPrimaryTransform,
)
from vulnflow.transforms.inferences import EventInference, TagInference

# Project root data directory
DATA_DIR = Path(__file__).parent.parent / "data"

# NVD base URL
NVD_BASE_URL = "https://nvd.nist.gov/feeds/json/cve/2.0"

# Default transforms for building cache
DEFAULT_TRANSFORMS = [
    CVSSFlattenTransform(),
    CWEPrimaryTransform(),
    CPEPrimaryTransform(),
    EventInference(),
    CVDStateTransform(),
    TagInference(),
]


def download_nvd_file(
    filename: str, force: bool = False, verbose: bool = True
) -> Path:
    """Download a single NVD JSON file.

    Args:
        filename: NVD filename (e.g., "nvdcve-2.0-2024.json" or "nvdcve-2.0-Modified.json")
        force: If True, re-download even if file exists
        verbose: If True, print progress messages

    Returns:
        Path to downloaded JSON file

    Example:
        >>> path = download_nvd_file("nvdcve-2.0-2024.json")
        >>> ds = VulnDataset.from_nvd(path)
    """
    DATA_DIR.mkdir(exist_ok=True)

    json_path = DATA_DIR / filename
    gz_path = DATA_DIR / f"{filename}.gz"

    # Check if already exists
    if json_path.exists() and not force:
        if verbose:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            count = data.get("totalResults", "unknown")
            print(f"[SKIP] {filename} already exists ({count} CVEs)")
        return json_path

    # Download
    url = f"{NVD_BASE_URL}/{filename}.gz"
    if verbose:
        print(f"[DOWNLOAD] {filename}...", end=" ", flush=True)

    try:
        urlretrieve(url, gz_path)
    except Exception as e:
        if verbose:
            print(f"[ERROR] {e}")
        raise

    # Extract
    with gzip.open(gz_path, "rb") as f_in:
        with open(json_path, "wb") as f_out:
            f_out.write(f_in.read())

    # Cleanup
    gz_path.unlink()

    # Report
    if verbose:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        count = data.get("totalResults", "unknown")
        size_mb = json_path.stat().st_size / 1024 / 1024
        print(f"[OK] {count} CVEs ({size_mb:.1f} MB)")

    return json_path


def download_modified(force: bool = False, verbose: bool = True) -> Path:
    """Download NVD Modified feed (last 8 days).

    Args:
        force: If True, re-download even if file exists
        verbose: If True, print progress messages

    Returns:
        Path to nvdcve-2.0-Modified.json

    Example:
        >>> path = download_modified()
        >>> ds = VulnDataset.from_nvd(path)
    """
    return download_nvd_file("nvdcve-2.0-Modified.json", force=force, verbose=verbose)


def download_recent(force: bool = False, verbose: bool = True) -> Path:
    """Download NVD Recent feed (recent years, ~240K CVEs).

    Args:
        force: If True, re-download even if file exists
        verbose: If True, print progress messages

    Returns:
        Path to nvdcve-2.0-Recent.json

    Example:
        >>> path = download_recent()
        >>> ds = VulnDataset.from_nvd(path)
    """
    return download_nvd_file("nvdcve-2.0-Recent.json", force=force, verbose=verbose)


def download_year(year: int, force: bool = False, verbose: bool = True) -> Path:
    """Download NVD data for a specific year.

    Args:
        year: Year to download (2002-2025)
        force: If True, re-download even if file exists
        verbose: If True, print progress messages

    Returns:
        Path to nvdcve-2.0-{year}.json

    Example:
        >>> path = download_year(2024)
        >>> ds = VulnDataset.from_nvd(path)
    """
    if year < 2002 or year > 2025:
        raise ValueError(f"Year must be between 2002 and 2025, got {year}")

    return download_nvd_file(f"nvdcve-2.0-{year}.json", force=force, verbose=verbose)


def download_year_range(
    start_year: int, end_year: int, force: bool = False, verbose: bool = True
) -> list[Path]:
    """Download NVD data for a range of years.

    Args:
        start_year: First year to download (2002-2025)
        end_year: Last year to download (2002-2025)
        force: If True, re-download even if files exist
        verbose: If True, print progress messages

    Returns:
        List of paths to downloaded JSON files

    Example:
        >>> paths = download_year_range(2020, 2024)
        >>> datasets = [VulnDataset.from_nvd(p) for p in paths]
    """
    if start_year < 2002 or start_year > 2025:
        raise ValueError(f"start_year must be between 2002 and 2025, got {start_year}")
    if end_year < 2002 or end_year > 2025:
        raise ValueError(f"end_year must be between 2002 and 2025, got {end_year}")
    if start_year > end_year:
        raise ValueError(
            f"start_year ({start_year}) must be <= end_year ({end_year})"
        )

    if verbose:
        print(f"Downloading NVD data for {start_year}-{end_year}...")

    paths = []
    for year in range(start_year, end_year + 1):
        path = download_year(year, force=force, verbose=verbose)
        paths.append(path)

    if verbose:
        print(f"\n[OK] Downloaded {len(paths)} files")

    return paths


def build_cache(
    sources: list[Path] | list[str],
    output_path: Path | str | None = None,
    transforms: list | None = None,
    verbose: bool = True,
) -> Path:
    """Build an enriched parquet cache from multiple NVD JSON files.

    Args:
        sources: List of paths to NVD JSON files
        output_path: Path to save parquet cache (default: data/nvd_cache.parquet)
        transforms: List of transforms to apply (default: DEFAULT_TRANSFORMS)
        verbose: If True, print progress messages

    Returns:
        Path to created parquet cache

    Example:
        >>> # Build cache from 2020-2024
        >>> paths = download_year_range(2020, 2024)
        >>> cache = build_cache(paths)
        >>> df = VulnDataset.from_parquet(cache).to_pandas()
    """
    if transforms is None:
        transforms = DEFAULT_TRANSFORMS

    if output_path is None:
        output_path = DATA_DIR / "nvd_cache.parquet"

    output_path = Path(output_path)

    if verbose:
        print(f"Building cache from {len(sources)} files...")
        print(f"  Transforms: {len(transforms)}")
        print(f"  Output: {output_path}")

    # Load all files
    datasets = []
    for source in sources:
        source = Path(source)
        if verbose:
            print(f"  Loading {source.name}...", end=" ", flush=True)
        ds = VulnDataset.from_nvd(str(source))
        if verbose:
            print(f"{len(ds)} CVEs")
        datasets.append(ds)

    # Merge datasets using parquet multi-file loading
    # First save each to temp parquet
    temp_dir = DATA_DIR / ".temp_cache_build"
    temp_dir.mkdir(exist_ok=True)

    temp_files = []
    for i, ds in enumerate(datasets):
        temp_path = temp_dir / f"temp_{i}.parquet"
        ds.to_parquet(temp_path)
        temp_files.append(temp_path)

    # Load merged dataset with upsert
    if verbose:
        print(f"\n  Merging {len(temp_files)} datasets...")
    merged_ds = VulnDataset.from_parquet([str(p) for p in temp_files])

    # Cleanup temp files
    for temp_file in temp_files:
        temp_file.unlink()
    temp_dir.rmdir()

    if verbose:
        print(f"  Merged dataset: {len(merged_ds)} CVEs")
        print(f"\n  Applying {len(transforms)} transforms...")

    # Apply transforms
    enriched_ds = merged_ds.transform(transforms)

    # Save cache
    if verbose:
        print(f"\n  Saving cache to {output_path}...")
    enriched_ds.to_parquet(output_path)

    if verbose:
        size_mb = output_path.stat().st_size / 1024 / 1024
        print(f"\n[OK] Cache built: {len(enriched_ds)} CVEs ({size_mb:.1f} MB)")
        print(f"     Columns: {len(enriched_ds.columns)}")

    return output_path


def get_available_years() -> list[int]:
    """Get list of years with available NVD data.

    Returns:
        List of years from 2002 to 2025
    """
    return list(range(2002, 2026))


def list_downloaded_files() -> dict[str, Path]:
    """List all downloaded NVD files in data directory.

    Returns:
        Dict mapping filename to Path

    Example:
        >>> files = list_downloaded_files()
        >>> for name, path in files.items():
        ...     print(f"{name}: {path.stat().st_size / 1024 / 1024:.1f} MB")
    """
    if not DATA_DIR.exists():
        return {}

    files = {}
    for json_file in DATA_DIR.glob("nvdcve-2.0-*.json"):
        files[json_file.name] = json_file

    return files
