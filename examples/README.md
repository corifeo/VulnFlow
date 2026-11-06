# Examples

## Setup

```bash
uv sync --extra notebook    # install dependencies, including Jupyter
uv run jupyter lab          # launch notebooks
```

Set `NVD_API_KEY` for faster API queries (10x rate limit):
```bash
export NVD_API_KEY=your-key-here
```

## Notebooks

### Tier 1: Getting Started

| Notebook | Persona | API Surface |
|----------|---------|-------------|
| `01-first-look.ipynb` | New user, first 5 minutes | `CacheBuilder.build()`, `to_pandas()` |
| `02-build-library.ipynb` | Building a persistent cache | `CacheBuilder.ensure()`, `from_parquet([base, delta])` |
| `03-weekly-triage.ipynb` | Security analyst, Monday triage | `CacheBuilder.build()`, `VulnDataset.merge()`, composite risk scoring |

### Tier 2: Advanced Workflows

| Notebook | Persona | API Surface |
|----------|---------|-------------|
| `04-patch-tuesday.ipynb` | Patch management analyst | `NVDAPIExtractor`, explicit transforms, CPE filtering |
| `05-threat-landscape.ipynb` | Security manager, quarterly report | `VulnDataset.from_parquet()`, `count_by()`, analytics |
| `06-custom-pipeline.ipynb` | Developer building ETL pipelines | `VulnPipeline`, `merge()`, `then()`, `PipelineResult` |

## Scripts

| Script | Persona | API Surface |
|--------|---------|-------------|
| `scripts/refresh_cache.py` | DevOps, cron job | `CacheBuilder.ensure()` |
| `scripts/critical_alert.py` | SOC analyst, alerting | `NVDAPIExtractor`, `ds.filter()`, `to_json()` |
| `scripts/etl_pipeline.py` | Platform engineer, CI/CD | `VulnPipeline`, `NVDExtractor`, `ParquetLoader` |

Run scripts with:
```bash
uv run examples/scripts/refresh_cache.py
uv run examples/scripts/critical_alert.py
uv run examples/scripts/etl_pipeline.py
```

## Troubleshooting

**NVD API rate limited (HTTP 403)**
Set `NVD_API_KEY` env var. Request a key at https://nvd.nist.gov/developers/request-an-api-key

**`ModuleNotFoundError: vulnflow`**
Run from the project root with `uv run`.

**Notebooks fail on restart-and-run-all**
Notebooks that use the NVD API (`04-patch-tuesday`) require network access.
Notebooks using `CacheBuilder.ensure()` (`02`, `05`) skip downloads when the cache is fresh.

**Empty charts / zero records**
The Recent feed covers ~8 days; the Modified feed covers ~2 hours of changes.
If datasets are sparse, use `CacheBuilder(years=[2024])` for richer data.
