"""
Download sample NVD data for notebook examples.

Downloads a subset of real NVD data to examples/data/ directory.
This data is NOT committed to git (too large), but needed for realistic demos.

Usage:
    uv run python examples/setup_data.py
"""

import gzip
import json
from pathlib import Path
from urllib.request import urlretrieve


def download_nvd_sample():
    """Download NVD Modified feed (smaller, real data)."""
    # Save to project root data/ directory
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)

    print("Downloading NVD Modified feed...")
    print("(This contains CVEs modified in the last 8 days - ~2-5K records)")

    url = "https://nvd.nist.gov/feeds/json/cve/2.0/nvdcve-2.0-Modified.json.gz"
    gz_path = data_dir / "nvdcve-2.0-Modified.json.gz"
    json_path = data_dir / "nvdcve-2.0-Modified.json"

    if json_path.exists():
        print(f"[OK] Already exists: {json_path}")
        with open(json_path, encoding='utf-8') as f:
            data = json.load(f)
        print(f"  Records: {data.get('totalResults', 'unknown')}")
        return

    # Download
    print(f"  Downloading to: {gz_path}")
    urlretrieve(url, gz_path)

    # Extract
    print(f"  Extracting to: {json_path}")
    with gzip.open(gz_path, "rb") as f_in:
        with open(json_path, "wb") as f_out:
            f_out.write(f_in.read())

    # Cleanup
    gz_path.unlink()

    # Report
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)
    print(f"\n[OK] Download complete!")
    print(f"  File: {json_path}")
    print(f"  Size: {json_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  Records: {data.get('totalResults', 'unknown')}")


if __name__ == "__main__":
    download_nvd_sample()
