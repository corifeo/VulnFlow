# Changelog

All notable changes to this project will be documented in this file.

Versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Every user-visible change adds a line under `## Unreleased` in the same PR.

## 0.1.0 (2026-11-03)

First public release.

### Added
- Extractors: NVD JSON feeds, NVD REST API 2.0, MSRC CVRF, Parquet, JSON, pandas, polars
- Enrichers: CISA KEV, EPSS, CWE, CAPEC, NVD change history
- Inference and analytics: CVD event timelines, tags, STRIDE categories, CVD state and desiderata
- Loaders: Parquet, JSON, pandas, polars
- `VulnDataset`, `VulnPipeline` and `CacheBuilder` entry points
- Concrete stages importable from `vulnflow.extractors` and `vulnflow.loaders`

### Fixed
- KEV dates were stored as strings and silently dropped from CVD comparisons, so zero-day and attack flags were wrong for every KEV CVE. `kev_date_added` and `kev_due_date` are now datetimes
- Null dates reloaded from Parquet were counted as events that had happened
- `EventInference` never ran tier 3 by default, leaving most CVEs without a vendor date. The default threshold is now 0.6, matching its documentation
- `EventInference` crashed on datasets reloaded from Parquet
- `CVDStateTransform` with no timestamp columns reported `vfdpxa` for every CVE. It now raises `TransformError`
- A dependency missing from both the run and the dataset now logs a warning
- `to_pandas()` returned naive or string date columns under pandas 3. Date columns are now always UTC

### Changed
- Licensed under MIT
- `ipywidgets` and `matplotlib` moved to the `notebook` extra
- `vulnflow.__version__` is read from package metadata

### Removed
- `ExploitDBEnricher` and `CacheBuilder`'s `exploit=` option, which never returned data
