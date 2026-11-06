<h1 align="center">vulnflow</h1>

<p align="center">
  <b>Fast, boring ETL for public vulnerability data. NVD, CISA KEV, EPSS, CWE, CAPEC and Microsoft's Patch Tuesday feed, normalized into one columnar dataset for pandas, polars or Parquet.</b>
</p>

<p align="center">
  <img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white">
  <img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-green">
  <img alt="Typed" src="https://img.shields.io/badge/types-py.typed-blue">
  <a href="https://github.com/corifeo/vulnflow/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/corifeo/vulnflow/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="Data: NVD · KEV · EPSS · CWE · CAPEC · MSRC" src="https://img.shields.io/badge/data-NVD%20%C2%B7%20KEV%20%C2%B7%20EPSS%20%C2%B7%20CWE%20%C2%B7%20CAPEC%20%C2%B7%20MSRC-orange">
</p>

<p align="center">
  <a href="#installation">Install</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#pipelines">Pipelines</a> ·
  <a href="#case-study-log4shell">Log4Shell case study</a> ·
  <a href="#what-the-dates-mean">What the dates mean</a> ·
  <a href="#sources-and-outputs">Sources</a>
</p>

<p align="center">
  <img src=".github/assets/pipeline.svg" alt="Public feeds are extracted into one columnar dataset, enriched, inferred and computed, then loaded into pandas, polars, Parquet or JSON" width="100%">
</p>

If you've spent half a notebook cleaning CVE feeds before doing any analysis, vulnflow is for you. It moves public vulnerability data and makes it sane:

- **One line to a dataframe.** `CacheBuilder(recent_only=True).build().to_pandas()` pulls the last week of NVD with KEV, EPSS, CWE and flattened CVSS in about ten seconds.
- **Columnar all the way.** Data lives in NumPy arrays, not record objects. CVD event inference and state for 83,000 CVEs takes under three seconds.
- **Explicit pipelines.** Extractors, transforms and loaders run exactly as you list them, in dependency order. A missing dependency is reported, not silently skipped.
- **Typed and tested.** `py.typed`, `mypy --strict`, about 1,200 tests on Python 3.11 to 3.14.

The pipeline core has no risk scoring and no opinions. `vulnflow.helpers.explore` adds optional helpers such as `risk_priority()`; treat them as a starting point, not a policy.

## Installation

vulnflow isn't on PyPI yet. Install it from GitHub:

```bash
uv add git+https://github.com/corifeo/vulnflow
```

```bash
pip install git+https://github.com/corifeo/vulnflow
```

From source, with the example notebooks and Jupyter:

```bash
git clone https://github.com/corifeo/vulnflow && cd vulnflow
uv sync --extra notebook
```

Requires Python 3.11+. HTTP responses are cached under `.cache/` in the working directory.

## Quick start

```python
from vulnflow.helpers import CacheBuilder

df = CacheBuilder(recent_only=True).build().to_pandas()
```

That's NVD's Recent feed, the CVEs published in the last eight days, with KEV, EPSS, CWE and CPE already joined and CVSS flattened. On 2026-11-06 it returned 3,681 CVEs:

| Field | Filled |
|---|---|
| CVSS base score | 91% |
| EPSS probability | 97% |
| CWE | 95% |
| CPE vendor | 11% |

The CPE gap is real: NVD's analysis backlog means most new CVEs have no product data yet. [`scripts/readme_numbers.py`](scripts/readme_numbers.py) regenerates the data figures in this README; a test pins the Log4Shell values.

For a persistent library, build once and reload from Parquet while it's fresh:

```python
from vulnflow.helpers import CacheBuilder

ds = CacheBuilder(years=[2024, 2025]).ensure("data/library.parquet", max_age="7d")
```

## Pipelines

```python
from vulnflow import VulnPipeline
from vulnflow.extractors import NVDExtractor
from vulnflow.helpers import CacheBuilder
from vulnflow.loaders import ParquetLoader
from vulnflow.transforms.computed import CVDStateTransform
from vulnflow.transforms.enrichers import KEVEnricher
from vulnflow.transforms.flatten import KEVFlattenTransform
from vulnflow.transforms.inferences import EventInference

pipeline = VulnPipeline(
    extractors=[NVDExtractor(CacheBuilder.feed_url(2025))],
    transformers=[KEVEnricher(), KEVFlattenTransform(), EventInference(), CVDStateTransform()],
    loaders=[ParquetLoader("nvd-2025.parquet")],
)
result = pipeline.run()

result.errors              # failed stages; the run continues past them unless strict=True
result.dataset.to_pandas()
```

Transforms run in phases computed from their `depends_on`. Pipelines compose: `extract.merge(enrich)` combines stages, `VulnPipeline.chain([...])` merges a list. `VulnDataset.transform()` with no arguments runs the registry defaults, for notebooks.

## Case study: Log4Shell

```python
from vulnflow import VulnPipeline
from vulnflow.extractors import NVDAPIExtractor
from vulnflow.transforms.computed import CVDStateTransform
from vulnflow.transforms.enrichers import KEVEnricher
from vulnflow.transforms.flatten import KEVFlattenTransform
from vulnflow.transforms.inferences import EventInference

result = VulnPipeline(
    extractors=[NVDAPIExtractor(cve_id="CVE-2021-44228")],
    transformers=[KEVEnricher(), KEVFlattenTransform(), EventInference(), CVDStateTransform()],
).run()
row = result.dataset.to_pandas().iloc[0]

row.cvd_state              # 'VFdPXA': vendor aware, fix ready, public, exploit, attacks
row.is_zero_day_attack     # True: in KEV before the earliest vendor date vulnflow found
row.public_to_attack_days  # -0.43: CISA listed it the day NVD published it
```

[`tests/integration/test_log4shell.py`](tests/integration/test_log4shell.py) pins these values against a recorded NVD record.

Across the 345 CVEs published in 2024 and 2025 that are in KEV, 86 (25%) were listed on or before the day NVD published them. That figure needs only two dates every source agrees on, which brings us to the caveat.

## What the dates mean

CVD analytics compare six event dates: vendor aware (V), fix ready (F), fix deployed (D), public (P), exploit public (X) and attacks seen (A).

- **Only P and A are source dates.** P is NVD's publication date and A is the KEV listing date. Public feeds don't say when a vendor learned about a bug.
- **V, F and X are inferred** from NVD reference tags and dates in reference URLs, so they can be wrong. Log4Shell's V comes out as 11 December 2021; Apache actually knew in November.
- **Some events are filled in from others.** With no exploit date, X takes the KEV date; with no vendor date but a fix date, V takes the fix date. D (fix deployed) is never inferred, so the state always shows `d`.
- **No V means unknown, not unaware.** It shows as `v` in the state, and zero-day flags stay `False` rather than guessing.
- **Date-only sources sort first.** KEV and URL dates are midnight UTC, so a same-day KEV listing counts as before a 10:15 publication. A KEV date means attacks were already seen by that day.

`public_to_attack_days <= 0` is the most robust signal here, because it uses only P and A.

## Sources and outputs

| Stage | Available |
|---|---|
| Extract | NVD JSON feeds (files or URLs), NVD REST API 2.0, MSRC CVRF (Patch Tuesday), Parquet, JSON, pandas, polars |
| Enrich | CISA KEV, EPSS, CWE, CAPEC, NVD change history |
| Flatten | CVSS (v2, v3.x, v4), CPE, CWE, CAPEC, KEV, EPSS, STRIDE, MSRC |
| Infer | CVD event timestamps, tags, STRIDE categories |
| Compute | CVD state, zero-day and exposure flags, desiderata, time deltas |
| Load | Parquet, JSON, pandas, polars |

Extractors live in `vulnflow.extractors`, loaders in `vulnflow.loaders`, and transforms in `vulnflow.transforms.enrichers`, `.flatten`, `.inferences` and `.computed`.

## Examples

| | |
|---|---|
| [`01-first-look.ipynb`](examples/01-first-look.ipynb) | the first five minutes: Recent feed to pandas |
| [`02-build-library.ipynb`](examples/02-build-library.ipynb) | a persistent Parquet library with incremental updates |
| [`03-weekly-triage.ipynb`](examples/03-weekly-triage.ipynb) | a Monday triage of what changed this week |
| [`04-patch-tuesday.ipynb`](examples/04-patch-tuesday.ipynb) | NVD API queries and Microsoft Patch Tuesday |
| [`05-threat-landscape.ipynb`](examples/05-threat-landscape.ipynb) | trends and charts across 2024 and 2025 |
| [`06-custom-pipeline.ipynb`](examples/06-custom-pipeline.ipynb) | building and composing `VulnPipeline`s |
| [`examples/scripts/`](examples/scripts/) | cron refresh, a KEV alert, a CI pipeline |

Setup and troubleshooting: [`examples/README.md`](examples/README.md).

## Related

[vulnstate](https://github.com/corifeo/vulnstate) models a single CVE's disclosure lifecycle as a state machine. vulnflow is the data pipeline: it computes the same CVD state at scale from public feeds.

## Development

```bash
uv sync
uv run pytest
uv run ruff check src/ tests/
uv run mypy src/ --strict
uv run black --check src/ tests/
```

Issues and PRs are welcome, especially new extractors and enrichers. Each one is a small class; see the existing ones in `src/vulnflow/`.

## Citation

To cite vulnflow, use **Cite this repository** on GitHub ([`CITATION.cff`](CITATION.cff)).

## Data sources and notices

vulnflow downloads data at run time from its publishers; it ships none of it. Each source's terms apply to what you download.

- This product uses data from the NVD API but is not endorsed or certified by the NVD.
- CISA Known Exploited Vulnerabilities catalog, from the Cybersecurity and Infrastructure Security Agency.
- EPSS scores from FIRST and Empirical Security.
- CWE and CAPEC are trademarks of The MITRE Corporation.
- Microsoft Security Response Center CVRF API.

## License

[MIT](LICENSE)
