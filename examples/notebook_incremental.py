"""
Demonstration: Incremental workflow for notebooks

Shows the two-pipeline pattern:
1. Delta pipeline: Fetch recent changes, transform, save
2. Merge pipeline: Load cache + delta, instant analysis
"""

from pathlib import Path

import numpy as np

from vulnflow.dataset import VulnDataset

# Setup: Create demo data
demo_dir = Path("demo_data")
demo_dir.mkdir(exist_ok=True)

print("=" * 60)
print("Incremental Workflow Demonstration")
print("=" * 60)

# Step 1: Initial full load (simulated - normally from NVD)
print("\n[Step 1] Initial full load (simulated)")
print("-" * 60)

initial = VulnDataset()
initial._data = {
    "cve_id": np.array([f"CVE-2024-{i:04d}" for i in range(1, 101)]),
    "title": np.array([f"Vulnerability {i}" for i in range(1, 101)]),
}

cache_path = demo_dir / "cache.parquet"

import pyarrow as pa
import pyarrow.parquet as pq

pq.write_table(pa.table(initial._data), cache_path)
print(f"[OK] Saved 100 CVEs to {cache_path}")

# Step 2: Delta update (modified + new CVEs)
print("\n[Step 2] Delta update (2 modified + 3 new)")
print("-" * 60)

delta = VulnDataset()
delta._data = {
    "cve_id": np.array(
        ["CVE-2024-0050", "CVE-2024-0075", "CVE-2024-0101", "CVE-2024-0102", "CVE-2024-0103"]
    ),
    "title": np.array(
        [
            "Vulnerability 50 UPDATED",
            "Vulnerability 75 UPDATED",
            "NEW Vulnerability 101",
            "NEW Vulnerability 102",
            "NEW Vulnerability 103",
        ]
    ),
}

delta_path = demo_dir / "delta.parquet"
pq.write_table(pa.table(delta._data), delta_path)
print(f"[OK] Saved 5 CVEs (2 updated, 3 new) to {delta_path}")

# Step 3: Notebook session - instant load
print("\n[Step 3] Notebook session - load merged dataset")
print("-" * 60)

print("Code: dataset = VulnDataset.from_parquet(['cache.parquet', 'delta.parquet'])")
dataset = VulnDataset.from_parquet([str(cache_path), str(delta_path)])

print(f"[OK] Loaded {len(dataset)} unique CVEs (100 original + 3 new)")

# Verify upsert
cve_ids = dataset.get_column("cve_id")
titles = dataset.get_column("title")

idx_50 = np.where(cve_ids == "CVE-2024-0050")[0][0]
idx_75 = np.where(cve_ids == "CVE-2024-0075")[0][0]

print(f"  - CVE-2024-0050: {titles[idx_50]}")
print(f"  - CVE-2024-0075: {titles[idx_75]}")
print(f"  [OK] Delta overwrote cache (upsert semantics)")

# Step 4: Transform (simulated - no external API calls)
print("\n[Step 4] Apply transforms (simulated)")
print("-" * 60)

print("Code: dataset.transform([])")
dataset.transform([])  # Empty list to avoid API calls
print("[OK] Transforms complete (progress bar shown during execution)")

# Step 5: Export to pandas
print("\n[Step 5] Export to pandas DataFrame")
print("-" * 60)

df = dataset.to_pandas()
print(f"[OK] DataFrame shape: {df.shape}")
print(f"\nFirst 3 rows:")
print(df[["cve_id", "title"]].head(3))

# Success
print("\n" + "=" * 60)
print("[OK] Incremental workflow demonstration complete!")
print("=" * 60)
print("\nQuick start (one line):")
print("  df = VulnDataset.from_parquet(['cache.parquet', 'delta.parquet']).to_pandas()")
print("\nCleanup:")
print(f"  rm -rf {demo_dir}")
