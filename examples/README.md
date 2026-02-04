# VulnFlow Jupyter Notebook Examples

This directory contains Jupyter notebooks demonstrating key VulnFlow workflows for data science.

## Notebooks

### 01-quick-start.ipynb
**Goal**: Get started in 30 seconds

Demonstrates:
- Load NVD data directly to pandas (one-liner)
- Basic data exploration
- Save to parquet for fast reloading

**Use when**: First time using VulnFlow, want to explore data quickly

### 02-incremental-workflow.ipynb
**Goal**: Handle large datasets efficiently (300K+ CVEs)

Demonstrates:
- Cache + delta pattern for incremental updates
- Upsert semantics (delta overwrites cache)
- Fast notebook startup (~0.2s for 300K records)
- Production workflow recommendations

**Use when**: Working with full NVD dataset, want fast iteration

### 03-transforms-and-enrichment.ipynb
**Goal**: Add computed fields and enrichments

Demonstrates:
- Computed transforms (CVD state, desiderata)
- Event inference from timestamps/references
- Progress bars in Jupyter
- Caching enriched data
- Combining incremental + transforms

**Use when**: Need enriched data (CVD timeline, quality flags)

## Running the Notebooks

### Option 1: Jupyter Lab (Recommended)

```bash
# Install dependencies (including ipywidgets for progress bars)
uv add jupyter jupyterlab ipywidgets

# Launch
uv run jupyter lab

# Navigate to examples/ directory
```

### Option 2: Jupyter Notebook

```bash
# Install dependencies (including ipywidgets for progress bars)
uv add jupyter ipywidgets

# Launch
uv run jupyter notebook examples/
```

### Option 3: VS Code

1. Install Python extension
2. Open any `.ipynb` file
3. Select kernel: `.venv/Scripts/python.exe`
4. Run cells

## Quick Validation

Verify notebooks load correctly:

```bash
python -c "import json; \
  json.load(open('examples/01-quick-start.ipynb')); \
  json.load(open('examples/02-incremental-workflow.ipynb')); \
  json.load(open('examples/03-transforms-and-enrichment.ipynb')); \
  print('All notebooks valid')"
```

## Example Scripts

### notebook_incremental.py
Python script demonstrating incremental workflow (no Jupyter required).

Run with: `uv run python examples/notebook_incremental.py`

## Common Patterns

### One-liner: NVD → pandas
```python
from vulnflow import VulnDataset
df = VulnDataset.from_nvd("data.json").to_pandas()
```

### Incremental: Cache + Delta
```python
df = VulnDataset.from_parquet(["cache.parquet", "delta.parquet"]).to_pandas()
```

### With Transforms
```python
from vulnflow.transforms.computed import CVDStateTransform

df = VulnDataset.from_nvd("data.json") \
    .transform([CVDStateTransform()]) \
    .to_pandas()
```

### Save Enriched Cache
```python
VulnDataset.from_nvd("data.json") \
    .transform([CVDStateTransform()]) \
    .to_parquet("enriched_cache.parquet")
```

## Performance Targets

- **Cache + Delta merge**: 300K + 2K records in ~0.2s
- **NVD JSON load**: 300K records in ~30s
- **Parquet load**: ~10x faster than JSON
- **Transform application**: Depends on transform complexity

## Data Sources

### NVD Data Feeds
- **Recent**: https://nvd.nist.gov/feeds/json/cve/2.0/nvdcve-2.0-Recent.json.gz
- **Modified**: https://nvd.nist.gov/feeds/json/cve/2.0/nvdcve-2.0-Modified.json.gz

### Workflow
1. One-time: Download Recent → Build cache
2. Every session: Download Modified → Build delta
3. Merge cache + delta → Analyze

## Troubleshooting

### "Module not found" errors
```bash
uv sync  # Ensure all dependencies installed
```

### Progress bars not showing or warning about ipywidgets
Progress bars use `tqdm.auto` which auto-detects Jupyter.

**Warning: "IProgress not found. Please update jupyter and ipywidgets"**
```bash
uv add ipywidgets  # Install Jupyter widgets
```

Then restart your Jupyter kernel (Kernel → Restart Kernel)

### Slow parquet loading
First load may be slower due to JIT compilation. Subsequent loads are fast.

## Next Steps

After completing these notebooks, see:
- `docs/ARCHITECTURE.md` - Understanding VulnFlow design
- `docs/PUBLIC-API.md` - Full API reference
- `tests/` - More usage examples
