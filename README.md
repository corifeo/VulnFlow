# VulnFlow


vulnflow is a high-performance data pipeline for ingesting, normalizing, and loading public vulnerability data.

It is **not** a vulnerability management product.
It does **not** prioritize vulnerabilities.
It does **not** tell you what to fix.

vulnflow exists to make vulnerability data usable, fast, and boring.

If you are tired of spending half your notebook cleaning CVE feeds before doing real analysis, vulnflow is for you.

---

## What VulnFlow does well

- Ingests public vulnerability data from multiple sources
- Normalizes inconsistent schemas into a single model
- Handles deduplication and enrichment cleanly
- Runs fast and scales predictably
- Plays well with notebooks and data lakes

VulnFlow is infrastructure. It moves data and makes it sane.

---

## What VulnFlow deliberately does not do

- No risk scoring
- No prioritization
- No asset context
- No dashboards
- No opinions

Those problems are important, but they are not solved by pretending data pipelines are decision engines.

---

## Why VulnFlow exists

Vulnerability data is messy by default:
- Fields are missing or overloaded
- Multiple sources disagree
- Performance collapses at scale

Most teams solve this by writing glue code. Over and over again.

VulnFlow exists so you don’t have to.

---

## Architecture overview

VulnFlow is built around three pluggable concepts:

- **Extractors**  
  Pull data from external sources

- **Transformers**  
  Normalize, enrich, and reshape data

- **Loaders**  
  Write data to data lakes or analytical stores

Pipelines are explicit. Nothing is hidden. You always know what ran and why.

---

## Installation

```bash
uv add vulnflow
```

## Quick Start

### For Data Scientists (Interactive/Notebooks)

```python
from vulnflow import VulnDataset

# Simplest - just get data into pandas
df = VulnDataset.from_nvd("nvd-2024/*.json").to_pandas()

# With enrichment and transforms
dataset = VulnDataset.from_nvd("nvd-2024/*.json")
dataset.enrich()
dataset.transform()
df = dataset.to_pandas()
```

### For Data Engineers (Production/ETL)

```python
from vulnflow import VulnPipeline, NVDExtractor, KEVEnricher, CVDStateTransform, ParquetLoader

# Declarative pipeline configuration
pipeline = VulnPipeline(
    extractors=[NVDExtractor("nvd-2024/*.json")],
    transformers=[KEVEnricher(), CVDStateTransform()],
    loaders=[ParquetLoader("output.parquet")]
)
result = pipeline.run()

# Compose pipelines
extract = VulnPipeline(extractors=[NVDExtractor("nvd-2024/*.json")])
enrich = VulnPipeline(transformers=[KEVEnricher()])
full = extract.then(enrich)
```

## Features

- **Extract**: NVD, CVE, OSV, Parquet, JSON, SQL
- **Transform**: KEV enrichment, EPSS scores, CVD state computation, Desiderata metrics
- **Load**: Polars, Pandas, Parquet, SQL, JSON

## License

VulnFlow is licensed under the Mozilla Public License 2.0.

You are free to use it, modify it, and build on it.
Changes to core files must be shared under the same license.

This project is open source, but it is not a free-for-all.