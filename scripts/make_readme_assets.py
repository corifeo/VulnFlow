"""Regenerate the README screenshots and charts in .github/assets/.

Also prints the figures README.md quotes. The hero uses NVD's live Recent
feed. The status screenshot and the KEV chart share one live build of all of
NVD (every yearly feed, a few minutes), saved to data/readme_library.parquet.
The Log4Shell case study uses the recorded fixtures in tests/fixtures/.

    uv run --with rich --extra notebook python scripts/make_readme_assets.py
"""

from __future__ import annotations

import io
import time
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rich import box
from rich.console import Console, RenderableType
from rich.measure import Measurement
from rich.table import Table
from rich.text import Text

from vulnflow import VulnDataset, VulnPipeline
from vulnflow.extractors import NVDExtractor
from vulnflow.helpers import CacheBuilder
from vulnflow.transforms.computed import CVDStateTransform
from vulnflow.transforms.enrichers import KEVEnricher
from vulnflow.transforms.flatten import KEVFlattenTransform
from vulnflow.transforms.inferences import EventInference

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / ".github" / "assets"
FIXTURES = ROOT / "tests" / "fixtures"

# Same palette and chart chrome as vulnstate's README assets
BLUE_RAMP = [
    "#cde2fb",
    "#9ec5f4",
    "#6da7ec",
    "#3987e5",
    "#256abf",
    "#184f95",
    "#0d366b",
]
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#f0efec",
        "ink2": "#c3c2b7",
        "muted": "#898781",
    },
}


def recording_console(width: int = 104) -> Console:
    return Console(
        record=True,
        width=width,
        file=io.StringIO(),
        force_terminal=True,
        color_system="truecolor",
    )


def save_svg(name: str, title: str, *renderables: RenderableType) -> None:
    # Size the terminal to the widest block so nothing wraps
    wide = recording_console(300)
    width = max(Measurement.get(wide, wide.options, r).maximum for r in renderables)
    console = recording_console(width)
    for renderable in renderables:
        console.print(renderable)
    console.save_svg(str(ASSETS / name), title=title)


def yes_no(flag: bool, bad_when_true: bool) -> Text:
    if flag:
        return Text("yes", style="bold red" if bad_when_true else "bold green")
    return Text("no", style="green" if bad_when_true else "red")


# ----------------------------------------------------------------- pipeline.svg


def make_pipeline_svg() -> None:
    """The quick start on today's Recent feed: what ran and what came out."""
    start = time.time()
    ds = CacheBuilder(recent_only=True, quiet=True).build()
    elapsed = time.time() - start
    df = ds.to_pandas()

    stages = Table(box=box.ROUNDED, header_style="bold cyan", title="What ran")
    stages.add_column("Stage", no_wrap=True)
    for name in ds.provenance["extractors"] + ds.provenance["transforms"]:
        stages.add_row(Text(name, style="magenta"))

    result = Table(box=box.ROUNDED, header_style="bold cyan", title="What you get")
    result.add_column("Field", no_wrap=True)
    result.add_column("Filled", justify="right", no_wrap=True)
    for label, column in [
        ("CVSS base score", "cvss_base_score"),
        ("EPSS probability", "epss_probability"),
        ("CWE", "cwe_id"),
        ("CPE vendor", "cpe_vendor"),
    ]:
        result.add_row(label, f"{df[column].notna().mean():.0%}")
    in_kev = int(df["is_kev"].fillna(False).astype(bool).sum())
    result.add_row(Text("In CISA KEV", style="bold"), f"{in_kev}")

    header = Text.assemble(
        (">>> ", "dim"),
        "CacheBuilder(recent_only=True).build().to_pandas()",
        (f"   # NVD Recent feed, {date.today():%d %b %Y}", "dim"),
    )
    footer = Text.assemble(
        (f"{len(df):,} CVEs", "bold"), f" × {df.shape[1]} columns in {elapsed:.0f}s"
    )
    tables = Table.grid(padding=(0, 4))
    tables.add_row(stages, result)
    save_svg("pipeline.svg", "vulnflow · quick start", header, Text(""), tables, footer)
    cpe = df["cpe_vendor"].notna().mean()
    print(
        f"Recent feed: {len(df):,} CVEs, {df.shape[1]} columns, {elapsed:.0f}s, CPE {cpe:.0%}"
    )


# ---------------------------------------------------------------- nvd_status.svg

STATUSES = [  # (status, what it means)
    ("analyzed", "NVD has analyzed it"),
    ("modified", "changed since NVD analyzed it"),
    ("awaiting", "not analyzed yet"),
    ("deferred", "NVD doesn't plan to analyze it"),
    ("rejected", "withdrawn by its CNA"),
]


def build_full_library() -> VulnDataset:
    """All of NVD in one step, then the status question from the README."""
    start = time.time()
    ds = CacheBuilder(quiet=True).build()
    built = time.time() - start
    path = ROOT / "data" / "readme_library.parquet"
    path.parent.mkdir(exist_ok=True)
    ds.to_parquet(path)
    start = time.time()
    VulnDataset.from_parquet(str(path))
    reload = time.time() - start

    counts = ds.get_status_distribution()
    total = sum(counts.values())
    status = Table(
        box=box.ROUNDED, header_style="bold cyan", title="NVD status, every CVE"
    )
    for col, justify in [
        ("Status", "left"),
        ("CVEs", "right"),
        ("Share", "right"),
        ("Meaning", "left"),
    ]:
        status.add_column(col, justify=justify, no_wrap=True)  # type: ignore[arg-type]
    for name, meaning in STATUSES:
        n = counts.get(name, 0)
        style = "bold yellow" if name == "deferred" else "magenta"
        status.add_row(Text(name, style=style), f"{n:,}", f"{n / total:.0%}", meaning)

    deferred = ds.to_pandas(analyzed=False, modified=False, awaiting=False)
    years = deferred["nvd_published_at"].dt.year
    by_era = Table(box=box.ROUNDED, header_style="bold cyan", title="Deferred, by year")
    by_era.add_column("Published", no_wrap=True)
    by_era.add_column("CVEs", justify="right", no_wrap=True)
    for year, n in years.value_counts().sort_index().items():
        by_era.add_row(str(year), f"{n:,}")

    header = Text.assemble(
        (">>> ", "dim"),
        'ds = CacheBuilder().ensure("data/nvd.parquet", max_age="1d")',
        (f"   # all of NVD, {date.today():%d %b %Y}", "dim"),
        "\n",
        (">>> ", "dim"),
        "ds.get_status_distribution()",
    )
    footer = Text.assemble(
        (f"{len(ds):,} CVEs", "bold"),
        f" · built in {built:.0f}s, reloads in {reload:.0f}s",
    )
    tables = Table.grid(padding=(0, 4))
    tables.add_row(status, by_era)
    save_svg(
        "nvd_status.svg",
        "vulnflow · the full library",
        header,
        Text(""),
        tables,
        footer,
    )
    print(
        f"All of NVD: {len(ds):,} CVEs, built {built:.0f}s, reload {reload:.0f}s; statuses {counts}"
    )
    print(f"Deferred by year: {years.value_counts().sort_index().to_dict()}")
    return ds


# --------------------------------------------------------- kev_timing_*.png

BUCKETS = [  # (label, upper bound on public_to_attack_days)
    ("Before\npublication", -1.0),
    ("Same day", 0.0),
    ("1–7\ndays", 7.0),
    ("8–30\ndays", 30.0),
    ("31–90\ndays", 90.0),
    ("91–365\ndays", 365.0),
    ("Over\na year", np.inf),
]


def make_kev_chart(ds: VulnDataset) -> None:
    """When CISA listed 2024-25 CVEs, relative to NVD publishing them."""
    # Same CVEs as the 2024 and 2025 yearly feeds the README snippet builds
    ids = ds.get_column("cve_id").astype(str)
    in_years = np.char.startswith(ids, "CVE-2024-") | np.char.startswith(
        ids, "CVE-2025-"
    )
    in_kev = np.array([bool(x) for x in ds.get_column("is_kev")])
    kev = ds.filter(in_kev & in_years)
    kev.transform([EventInference(), CVDStateTransform()], quiet=True)
    days = kev.get_column("public_to_attack_days").astype(float)

    counts, lower = [], -np.inf
    for _, upper in BUCKETS:
        counts.append(int(((days > lower) & (days <= upper)).sum()))
        lower = upper
    early = counts[0] + counts[1]

    for mode, t in THEMES.items():
        fig, ax = plt.subplots(figsize=(8.4, 4.2), dpi=200)
        fig.patch.set_facecolor(t["surface"])
        ax.set_facecolor(t["surface"])
        x = np.arange(len(BUCKETS))
        colors = [BLUE_RAMP[5]] * 2 + [BLUE_RAMP[2]] * (len(BUCKETS) - 2)
        bars = ax.bar(x, counts, width=0.72, color=colors, linewidth=0)
        for bar, n in zip(bars, counts, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(counts) * 0.02,
                f"{n}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
                color=t["ink"],
            )
        ax.set_xticks(x, [label for label, _ in BUCKETS], color=t["ink2"], fontsize=9)
        ax.set_yticks([])
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel(
            "CISA KEV listing, relative to NVD publication",
            color=t["muted"],
            fontsize=9,
        )
        fig.suptitle(
            f"{len(kev)} KEV CVEs from 2024–25, by when CISA listed them",
            x=0.02,
            ha="left",
            color=t["ink"],
            fontsize=12,
            fontweight="bold",
        )
        fig.text(
            0.02,
            0.015,
            "Source: NVD 2024 and 2025 feeds + CISA KEV · computed by vulnflow",
            color=t["muted"],
            fontsize=7.5,
        )
        fig.tight_layout(rect=(0, 0.04, 1, 0.95))
        fig.savefig(ASSETS / f"kev_timing_{mode}.png", facecolor=t["surface"])
        plt.close(fig)
    print(
        f"KEV 2024-25: {len(kev)} CVEs, {early} ({early / len(kev):.0%}) on or before NVD"
    )


# ---------------------------------------------------------------- log4shell.svg

SOURCES = {  # inference confidence -> where the date came from
    0.95: "NVD publication / CISA KEV listing",
    0.80: "date in a reference URL",
    0.60: "reference tag (set to P)",
    0.50: "inferred from another event",
}


def make_log4shell_svg() -> None:
    """Log4Shell through the README chain, from the recorded NVD record and KEV entry."""
    result = VulnPipeline(
        extractors=[NVDExtractor(str(FIXTURES / "nvd_log4shell.json"))],
        transformers=[
            KEVEnricher(source=str(FIXTURES / "kev_log4shell.csv")),
            KEVFlattenTransform(),
            EventInference(),
            CVDStateTransform(),
        ],
    ).run(quiet=True, strict=True)
    assert result.dataset is not None
    row: dict[str, Any] = {
        c: result.dataset.get_column(c)[0] for c in result.dataset.columns
    }

    names = {"v": "V  vendor aware", "f": "F  fix ready", "p": "P  public"}
    names |= {"x": "X  exploit public", "a": "A  attacks", "d": "D  deployed"}
    timeline = Table(
        box=box.ROUNDED, header_style="bold cyan", title="Inferred timeline"
    )
    for col in ["Event", "Date (UTC)", "Where it came from"]:
        timeline.add_column(col, no_wrap=True)
    for e in "vfpxad":
        ts, conf = row.get(f"{e}_timestamp"), row.get(f"{e}_confidence")
        shown = "-" if ts is None else f"{ts:%d %b %Y %H:%M}"
        source = (
            "never inferred" if e == "d" else SOURCES.get(round(float(conf), 2), "")
        )
        if e in "pa" and conf is not None and round(float(conf), 2) == 0.95:
            source = "NVD publication" if e == "p" else "CISA KEV listing"
        timeline.add_row(Text(names[e], style="magenta"), shown, source)

    analytics = Table(
        box=box.ROUNDED, header_style="bold cyan", title="What vulnflow infers"
    )
    analytics.add_column("Metric", no_wrap=True)
    analytics.add_column("Result", justify="center", no_wrap=True)
    for name, cell in [
        ("State", Text(str(row["cvd_state"]), style="bold")),
        (
            "Zero-day exploit (X before V)",
            yes_no(bool(row["is_zero_day_exploit"]), True),
        ),
        ("Zero-day attack (A before V)", yes_no(bool(row["is_zero_day_attack"]), True)),
        (
            "Premature disclosure (P before F)",
            yes_no(bool(row["is_premature_disclosure"]), True),
        ),
        ("Attack before fix", yes_no(bool(row["is_attack_before_fix"]), True)),
        ("Desiderata met", Text(f"{int(row['desiderata_count'])} of 12", style="red")),
        (
            Text("Public → KEV", style="bold"),
            Text(f"{float(row['public_to_attack_days']):+.2f} days", style="bold red"),
        ),
    ]:
        analytics.add_row(name, cell)

    header = Text.assemble(
        (">>> ", "dim"),
        "Log4Shell  CVE-2021-44228  CVSS 10.0",
        ("   # recorded NVD record + CISA KEV entry", "dim"),
    )
    footer = Text.assemble(
        ("Caveat: ", "bold"),
        "V comes from a URL date. Apache actually knew in ",
        ("November", "yellow"),
    )
    save_svg(
        "log4shell.svg",
        "vulnflow · Log4Shell case study",
        header,
        Text(""),
        timeline,
        analytics,
        footer,
    )


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    make_log4shell_svg()
    make_pipeline_svg()
    make_kev_chart(build_full_library())
    print(f"wrote assets to {ASSETS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
