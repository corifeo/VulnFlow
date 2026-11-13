# Changelog

All notable changes to this project will be documented in this file.

Versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Every user-visible change adds a line under `## Unreleased` in the same PR.

## Unreleased

## 0.2.0 (2026-11-13)

### Added
- cvelistV5 as an alternative source: `CVEListExtractor` reads CVEProject/cvelistV5 release zips (URL, local zip or directory of `CVE-*.json`). `CacheBuilder(source="cvelist")` builds from the newest midnight zip, and `ensure(recent=True)` merges the latest hourly delta
- `CVEListExtractor(validate=True)` checks each record with cve-validation-library and adds `cve_valid` and `cve_validation_error`. Needs the optional `cvelist` extra
- Status filters accept `published` (cvelistV5 records), included by default
- `TriageLoader` writes a ranked SSVC triage queue (`vulntriage.queue/v1` JSON) with suggested Exploitation, Automatable and Technical Impact values and the evidence for each, plus `examples/scripts/triage_queue.py`
- `SECURITY.md` and `CONTRIBUTING.md`
- README: a full-library section (warehouse hand-off, NVD status counts) and a case study of 2026's CVE surge and NVD's growing analysis backlog, with generated charts

### Fixed
- `VulnDataset.merge()` and `CacheBuilder(include_modified=True)` duplicated every CVE present in more than one input, so a cache-plus-delta update counted modified CVEs twice. Merges now keep one row per `cve_id`. The newest `nvd_modified_at` wins; ties, and CVEs where any copy lacks a date, go to the later input. Multi-file `ParquetExtractor` uses the same rule
- Writing a dataset read back from Parquet dropped the UTC timezone from timestamp columns
- Re-running a flatten transform (EPSS, KEV, CVSS, CPE) on an already-flattened dataset kept the old values
- `ensure(recent=True)` refreshed EPSS and KEV only for changed CVEs. It now refreshes every row, rebuilds when the cache is older than the Modified feed's window (7.5 days), and writes the cache atomically
- A dropped download or a failed feed aborted the whole `build()`. Downloads retry on connection errors, and `build()` returns the feeds that succeeded with a warning, raising only when every feed fails

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
- MSRC `msrc_exploited` and `msrc_publicly_disclosed` were true for nearly every CVE, because MSRC's "Exploited:No" contains the word. They now read the field's value

### Changed
- Licensed under MIT
- `ipywidgets` and `matplotlib` moved to the `notebook` extra
- `vulnflow.__version__` is read from package metadata

### Removed
- `ExploitDBEnricher` and `CacheBuilder`'s `exploit=` option, which never returned data
