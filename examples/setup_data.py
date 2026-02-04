"""
Quick setup script for downloading NVD Modified feed.

For more control, use vulnflow_data library functions directly:
    from vulnflow_data import download_year_range, build_cache

Usage:
    uv run python examples/setup_data.py
"""

from vulnflow_data import download_modified

if __name__ == "__main__":
    print("Downloading NVD Modified feed...")
    print("(Last 8 days of CVE changes, ~2-5K records)")
    print()

    path = download_modified()

    print()
    print("[OK] Setup complete!")
    print(f"     File: {path}")
    print()
    print("Next steps:")
    print("  cd examples")
    print("  uv run jupyter lab")
