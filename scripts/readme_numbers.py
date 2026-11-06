"""Print the real-data numbers quoted in README.md.

Usage:
    uv run python scripts/readme_numbers.py

Downloads NVD's Recent feed, and rebuilds data/library.parquet (2024-2025)
when it is older than 7 days, which takes several minutes.
"""

from __future__ import annotations

from datetime import date

import numpy as np

from vulnflow.helpers import CacheBuilder
from vulnflow.transforms.computed import CVDStateTransform
from vulnflow.transforms.inferences import EventInference

FIELDS = {
    "CVSS base score": "cvss_base_score",
    "EPSS probability": "epss_probability",
    "CWE": "cwe_id",
    "CPE vendor": "cpe_vendor",
}


def fill_rates() -> None:
    df = CacheBuilder(recent_only=True, quiet=True).build().to_pandas()
    print(f"Recent feed on {date.today()}: {len(df):,} CVEs\n")
    print("| Field | Filled |\n|---|---|")
    for label, column in FIELDS.items():
        print(f"| {label} | {df[column].notna().mean():.0%} |")


def kev_before_publication() -> None:
    builder = CacheBuilder(years=[2024, 2025], quiet=True)
    ds = builder.ensure("data/library.parquet", max_age="7d")
    kev = ds.filter(np.array([bool(x) for x in ds.get_column("is_kev")]))
    kev.transform([EventInference(), CVDStateTransform()], quiet=True)
    early = int(np.sum(kev.get_column("public_to_attack_days") <= 0))
    share = early / len(kev)
    print(f"\nKEV CVEs from 2024-25: {len(kev)}; listed on or before NVD publication: ")
    print(f"{early} ({share:.0%})")


if __name__ == "__main__":
    fill_rates()
    kev_before_publication()
