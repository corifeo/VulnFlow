# VulnFlow

ETL Pipeline for Vulnerability Intelligence.

## Installation

```bash
uv add vulnflow
```

## Quick Start

```python
from vulnflow import VulnDataset

# Extract from NVD JSON files
dataset = VulnDataset.from_nvd("nvd-2024/*.json")

# Enrich with external data (KEV, EPSS)
dataset.enrich()

# Transform (CVD state, Desiderata metrics)
dataset.transform()

# Export to pandas DataFrame
df = dataset.to_pandas()
```

## Features

- **Extract**: NVD, CVE, OSV, Parquet, JSON, SQL
- **Transform**: KEV enrichment, EPSS scores, CVD state computation, Desiderata metrics
- **Load**: Polars, Pandas, Parquet, SQL, JSON

## License

MIT
