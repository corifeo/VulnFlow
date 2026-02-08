# VulnFlow Examples

## Choose Your Notebook

**Quick exploration?** → [01-quick-exploration.ipynb](01-quick-exploration.ipynb)
- NVD Modified feed (last 8 days, ~2-5K CVEs)
- Severity, critical CVEs, attack vectors

**Recent analysis?** → [02-recent-analysis.ipynb](02-recent-analysis.ipynb)
- Last 12 months (~50K CVEs)
- Monthly trends, score distributions, incremental updates

**Complete library?** → [03-complete-library.ipynb](03-complete-library.ipynb)
- Full NVD database (2002-present, ~330K CVEs)
- Yearly trends, status breakdown

**Copy-paste recipes?** → [04-cookbook.ipynb](04-cookbook.ipynb)
- Filtering, aggregation, vendor analysis
- CWE/CPE exploration, CVSS impact analysis
- Placeholder sections for EPSS, KEV, metadata, tags

---

## Setup

```bash
uv add jupyter jupyterlab
cd examples
uv run jupyter lab
```

All notebooks download data automatically. No manual setup needed.

---

## The CacheBuilder

Every notebook uses `CacheBuilder` - a one-line way to build vulnerability datasets:

```python
from vulnflow.helpers import CacheBuilder

# All years (2002-present)
ds = CacheBuilder().build()

# Specific years
ds = CacheBuilder(years=[2024, 2025]).build()

# Modified feed only (quick update)
ds = CacheBuilder(modified_only=True).build()

# Save
ds.to_parquet("data/library.parquet")
```

For incremental updates after initial build:

```python
from vulnflow import VulnDataset
from vulnflow.extractors.nvd import NVDExtractor
from vulnflow.helpers import CacheBuilder

delta = NVDExtractor(CacheBuilder.feed_url("modified")).extract()
ds = VulnDataset.from_parquet("data/library.parquet")
VulnDataset.merge([ds, delta]).to_parquet("data/library.parquet")
```

---

## Troubleshooting

**ImportError**: Run `uv sync` to update vulnflow.

**KeyError on column**: Dataset may need transforms. Call `ds.transform()` before `to_pandas()`.
