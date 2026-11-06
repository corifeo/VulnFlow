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
  <a href="#kev-timing-in-10-lines">Real-data KEV timing</a> ·
  <a href="#case-study-log4shell">Log4Shell case study</a> ·
  <a href="#what-the-dates-mean">What the dates mean</a> ·
  <a href="#pipelines">Pipelines</a> ·
  <a href="#sources-and-outputs">Sources</a>
</p>

<p align="center">
  <img src=".github/assets/pipeline.svg" alt="The quick start building all of NVD: the stages that ran and how complete each field came out" width="100%">
</p>

If you've spent half a notebook cleaning CVE feeds before doing any analysis, vulnflow is for you. It moves public vulnerability data and makes it sane:

- **One line to all of NVD.** `CacheBuilder().ensure("data/nvd.parquet").to_pandas()` builds every CVE NVD has published, joined with KEV, EPSS and CWE, in about two minutes, then reloads from Parquet in seconds.
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

df = CacheBuilder().ensure("data/nvd.parquet", max_age="1d").to_pandas()
```

That's every CVE NVD has published, 2002 to today, with KEV, EPSS, CWE and CPE joined and CVSS flattened. The first call downloads all the yearly feeds and builds the library; after that it reloads from Parquet until the file is a day old. The screenshot above is this call on 6 November 2026: 400,944 CVEs and 47 columns, built in 79 seconds and reloaded in 4. `to_pandas()` leaves out rejected CVEs by default.

For only the CVEs published in the last eight days, `CacheBuilder(recent_only=True).build()` takes seconds. Expect thinner data there: NVD's analysis backlog meant only 11% of them had product (CPE) data on the same day.

## KEV timing in 10 lines

How often does CISA list a CVE as exploited before NVD has even published it? Two years of NVD, joined with the KEV catalog:

```python
import numpy as np
from vulnflow.helpers import CacheBuilder
from vulnflow.transforms.computed import CVDStateTransform
from vulnflow.transforms.inferences import EventInference

ds = CacheBuilder(years=[2024, 2025]).ensure("data/library.parquet", max_age="7d")
kev = ds.filter(np.array([bool(x) for x in ds.get_column("is_kev")]))
kev.transform([EventInference(), CVDStateTransform()])

(kev.get_column("public_to_attack_days") <= 0).mean()   # 0.24
```

`ensure()` builds the library once and reloads it from Parquet while it's fresh.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset=".github/assets/kev_timing_dark.png">
  <img src=".github/assets/kev_timing_light.png" alt="354 KEV CVEs from 2024 and 2025 by when CISA listed them relative to NVD publication; 86 were listed on or before the publication day">
</picture>

Of the 354 CVEs from 2024 and 2025 in KEV, 86 (24%) were listed on or before the day NVD published them. That figure needs only two dates every source agrees on, which brings us to the caveat in [What the dates mean](#what-the-dates-mean). Every number above comes from real data; [`scripts/make_readme_assets.py`](scripts/make_readme_assets.py) regenerates it.

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

<p align="center">
  <img src=".github/assets/log4shell.svg" alt="Log4Shell's inferred disclosure timeline and the CVD analytics vulnflow derives from it" width="85%">
</p>

[`tests/integration/test_log4shell.py`](tests/integration/test_log4shell.py) pins these values against a recorded NVD record, and the screenshot comes from the same record.


## What the dates mean

CVD analytics compare six event dates: vendor aware (V), fix ready (F), fix deployed (D), public (P), exploit public (X) and attacks seen (A).

- **Only P and A are source dates.** P is NVD's publication date and A is the KEV listing date. Public feeds don't say when a vendor learned about a bug.
- **V, F and X are inferred** from NVD reference tags and dates in reference URLs, so they can be wrong. Log4Shell's V comes out as 11 December 2021; Apache actually knew in November.
- **Some events are filled in from others.** With no exploit date, X takes the KEV date; with no vendor date but a fix date, V takes the fix date. D (fix deployed) is never inferred, so the state always shows `d`.
- **No V means unknown, not unaware.** It shows as `v` in the state, and zero-day flags stay `False` rather than guessing.
- **Date-only sources sort first.** KEV and URL dates are midnight UTC, so a same-day KEV listing counts as before a 10:15 publication. A KEV date means attacks were already seen by that day.

`public_to_attack_days <= 0` is the most robust signal here, because it uses only P and A.

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

## Examples & notebooks

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
uv run --with rich --extra notebook python scripts/make_readme_assets.py   # regenerate README images
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
