# Jupyter Notebook Examples

Three notebooks + support library for VulnFlow data science workflows.

## Setup

### 1. Install dependencies

```bash
uv add jupyter jupyterlab ipywidgets
```

### 2. Download sample data

```bash
uv run python examples/setup_data.py
```

Downloads `nvdcve-2.0-Modified.json` (~2-5K CVEs) to `data/` directory.

### 3. Launch Jupyter

```bash
cd examples
uv run jupyter lab
```

## Support Library: `vulnflow_data.py`

Python module providing data management functions:

```python
from vulnflow_data import (
    download_modified,      # Download Modified feed (last 8 days)
    download_recent,        # Download Recent feed (~240K CVEs)
    download_year,          # Download specific year
    download_year_range,    # Download multiple years
    build_cache,            # Build enriched parquet cache
    list_downloaded_files,  # List downloaded files
)

# Download 2020-2024
paths = download_year_range(2020, 2024)

# Build enriched cache
cache = build_cache(paths, output_path="../data/my_cache.parquet")

# Load for analysis
df = VulnDataset.from_parquet(cache).to_pandas()
```

## Notebooks

### 01-quick-start.ipynb
**Goal:** Get started in 30 seconds

- Load NVD → flatten → pandas
- Filter critical vulnerabilities
- Explore severity distribution
- Cache to parquet

**Use when:** First time using VulnFlow, quick exploration

### 02-advanced-workflows.ipynb
**Goal:** Production patterns for large datasets

- Download data with support library
- Cache + delta incremental pattern
- Upsert semantics (delta overwrites cache)
- Fast notebook startup

**Use when:** Working with large datasets, need incremental updates

### 03-build-vulnerability-library.ipynb
**Goal:** Build complete CVE library from 2002-2025

- Download year ranges with `download_year_range()`
- Build enriched cache with all transforms
- CVD analytics (zero-days, fix lags, CVD states)
- Incremental updates with Modified feed

**Use when:** Building comprehensive vulnerability database

## Data Management

### NVD Feeds

- **Modified** (2-5K CVEs): Last 8 days of changes
- **Recent** (240K CVEs): All CVEs from recent years
- **Yearly** (varies): Historical data by year (2002-2025)

### Download Patterns

```python
# Quick start - Modified feed only
from vulnflow_data import download_modified
path = download_modified()

# Build library - multiple years
from vulnflow_data import download_year_range, build_cache
paths = download_year_range(2020, 2025)
cache = build_cache(paths)

# Complete library - all years
paths = download_year_range(2002, 2025)  # ~300K CVEs, ~150 MB
cache = build_cache(paths, output_path="../data/complete_library.parquet")
```

### Default Transforms

`build_cache()` applies these transforms by default:

```python
DEFAULT_TRANSFORMS = [
    CVSSFlattenTransform(),   # cvss_base_score, cvss_base_severity
    CWEPrimaryTransform(),    # cwe_id, cwe_name
    CPEFlattenTransform(),    # cpe_vendor, cpe_product
    EventInference(),         # v_timestamp, f_timestamp, confidence
    CVDStateTransform(),      # cvd_state, is_zero_day_exploit, fix_lag_days
    TagInference(),           # tag inference from references
]
```

Result: 50+ columns ready for analysis.

## Troubleshooting

### FileNotFoundError

Run setup script:
```bash
uv run python examples/setup_data.py
```

### KeyError: 'cvss_base_score'

Use flatten transforms:
```python
.transform([CVSSFlattenTransform(), CWEPrimaryTransform(), CPEFlattenTransform()])
```

Or use the support library:
```python
from vulnflow_data import build_cache
cache = build_cache([path])  # Applies transforms automatically
```

### "IProgress not found" warning

```bash
uv add ipywidgets
```

Then restart Jupyter kernel.

### Download fails

NVD URL structure: `https://nvd.nist.gov/feeds/json/cve/2.0/nvdcve-2.0-{year|Modified|Recent}.json.gz`

If downloads fail:
1. Check network connection
2. Verify NVD is accessible
3. Try forcing re-download: `download_year(2024, force=True)`

## Performance

- **Download**: ~1-5 MB/year, ~5 seconds/year
- **Build cache (2023-2025)**: ~100K CVEs, ~30 seconds
- **Build cache (2002-2025)**: ~300K CVEs, ~90 seconds
- **Load parquet**: ~1-3 seconds regardless of size
- **Cache + delta merge**: ~0.2 seconds for 300K + 2K

## File Organization

```
examples/
├── vulnflow_data.py                # Support library
├── setup_data.py                   # Quick setup script
├── 01-quick-start.ipynb           # Basic workflow
├── 02-advanced-workflows.ipynb    # Incremental patterns
└── 03-build-vulnerability-library.ipynb  # Complete library build

data/                               # Downloaded data (gitignored)
├── nvdcve-2.0-Modified.json        # Modified feed
├── nvdcve-2.0-2024.json            # Year-specific feeds
├── nvdcve-2.0-2023.json
├── nvd_library.parquet             # Enriched cache
└── nvd_delta.parquet               # Delta cache
```

## Example Workflows

### Daily analysis (with pre-built cache)

```python
from vulnflow import VulnDataset
from vulnflow_data import download_modified, build_cache

# Update delta
modified = download_modified(force=True)
build_cache([modified], output_path="../data/nvd_delta.parquet")

# Merge and analyze
df = VulnDataset.from_parquet([
    "../data/nvd_library.parquet",
    "../data/nvd_delta.parquet",
]).to_pandas()

# Analyze today's critical CVEs
critical = df[df['cvss_base_score'] >= 9.0]
```

### Build custom library

```python
from vulnflow_data import download_year_range, build_cache

# Download specific years
paths = download_year_range(2018, 2025)

# Build with custom transforms
from vulnflow.transforms.flatten import CVSSFlattenTransform

cache = build_cache(
    sources=paths,
    output_path="../data/custom_library.parquet",
    transforms=[CVSSFlattenTransform()],  # Only CVSS flattening
)
```

## Next Steps

- See notebook 01 for quick start
- See notebook 02 for incremental workflows
- See notebook 03 for complete library build
- Check `../tests/` for more code examples
- Read `../docs/ARCHITECTURE.md` for design details
