# Jupyter Notebook Examples

Three notebooks demonstrating VulnFlow for data science workflows.

## Setup

### 1. Install dependencies

```bash
uv add jupyter jupyterlab ipywidgets
```

### 2. Download sample data

```bash
uv run python examples/setup_data.py
```

This downloads `nvdcve-2.0-Modified.json` (~2-5K recent CVEs, ~5 MB) to `data/` (project root).

**Note:** Data files are gitignored (too large). The setup script downloads real NVD data.

### 3. Launch Jupyter

```bash
cd examples
uv run jupyter lab
```

## Notebooks

### 01-quick-start.ipynb

Load NVD → flatten → pandas. See CVSS scores, CWEs, severity distribution.

**Key pattern:**
```python
ds = VulnDataset.from_nvd("../data/nvdcve-2.0-Modified.json").transform([
    CVSSFlattenTransform(),  # cvss_base_score, cvss_base_severity, ...
    CWEPrimaryTransform(),   # cwe_id, cwe_name
    CPEPrimaryTransform(),   # cpe_vendor, cpe_product, cpe_version
])
df = ds.to_pandas()
```

### 02-incremental-workflow.ipynb

Cache + delta pattern for fast notebook startup.

**Key pattern:**
```python
# Build cache once (one-time cost)
VulnDataset.from_nvd("nvdcve-2.0-Recent.json") \
    .transform([...]) \
    .to_parquet("cache.parquet")

# Every session: merge cache + delta (~0.2s)
df = VulnDataset.from_parquet(["cache.parquet", "delta.parquet"]).to_pandas()
```

### 03-cvd-analytics.ipynb

Add CVD analytics: timelines, zero-day detection, fix lags.

**Key pattern:**
```python
ds = VulnDataset.from_nvd("../data/nvdcve-2.0-Modified.json").transform([
    CVSSFlattenTransform(),
    CWEPrimaryTransform(),
    CPEPrimaryTransform(),
    EventInference(),      # v_timestamp, f_timestamp, confidence scores
    CVDStateTransform(),   # cvd_state, is_zero_day_exploit, fix_lag_days
    TagInference(),        # tag inference from references
])
```

## Why Flatten Transforms?

NVD stores CVSS/CWE/CPE as nested JSON. Without flatten transforms, pandas DataFrames only get 13 base columns. With them, you get 40+ scalar columns ready for analysis.

## Data Files

```
data/                               # Project root data directory
├── nvdcve-2.0-Modified.json        # Downloaded by setup_data.py
├── cache.parquet                   # Created by notebooks (gitignored)
└── delta.parquet                   # Created by notebooks (gitignored)
```

All data files are gitignored (see root `.gitignore`).

## NVD Data Feeds

- **Modified** (2-5K CVEs): Last 8 days of changes - good for demos
- **Recent** (240K CVEs): All CVEs from recent years - full dataset
- **Source**: https://nvd.nist.gov/vuln/data-feeds

## Troubleshooting

### "FileNotFoundError: ../data/nvdcve-2.0-Modified.json"

Run setup script from project root:
```bash
uv run python examples/setup_data.py
```

This downloads NVD data to the `data/` directory.

### "KeyError: 'cvss_base_score'"

Apply flatten transforms:
```python
.transform([CVSSFlattenTransform(), CWEPrimaryTransform(), CPEPrimaryTransform()])
```

### "IProgress not found" warning

```bash
uv add ipywidgets
```

Then restart Jupyter kernel.

### Progress bars not showing

Already included with ipywidgets. If still not working:
1. Restart Jupyter kernel
2. Check ipywidgets is installed: `uv pip list | grep ipywidgets`

## Performance

- **NVD JSON load**: ~30s for 300K CVEs
- **Parquet load**: ~3s for 300K CVEs (10x faster)
- **Cache + delta merge**: ~0.2s for 300K + 2K
- **Flatten transforms**: ~1s for 2-5K CVEs

## Next Steps

After notebooks, see:
- `../tests/` - More code examples
- `../docs/ARCHITECTURE.md` - How VulnFlow works
- `../docs/ref/` - CVD model, CVSS, EPSS docs
