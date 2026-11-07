"""
Triage Loader - SSVC triage queue export for VulnTriage

Provides:
- TriageLoader: Writes a ranked queue of CVEs as JSON for an SSVC triage UI
- TRIAGE_SCHEMA: Schema identifier written into every export

Unlike the file loaders this is a *view*, not an archive: it filters, ranks
and caps the dataset, and adds SSVC evidence suggestions (exploitation,
automatable, technical impact) with the reason for each. Suggestions are
heuristics for an analyst to confirm, never decisions.

Layer: Load
Dependencies: dataset, utils.filtering
Used by: pipeline, user code
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from vulnflow.exceptions import LoaderError
from vulnflow.loaders.base import BaseLoader
from vulnflow.registry import Registry
from vulnflow.utils import json
from vulnflow.utils.filtering import _build_date_mask, _build_status_mask
from vulnflow.utils.nvd import to_utc_datetime

if TYPE_CHECKING:
    from vulnflow.dataset import VulnDataset

__all__ = ["TriageLoader", "TRIAGE_SCHEMA"]

TRIAGE_SCHEMA = "vulntriage.queue/v1"

_EXPLOIT_TAG = "Exploit"
_MAX_REFERENCES = 15
_TITLE_LENGTH = 110


# ============================================================================
# Small readers over one row
# ============================================================================


def _cell(dataset: VulnDataset, column: str, i: int) -> Any:
    """Value at row i, or None if the column is absent or the value is NaN."""
    arr = dataset.get_column_or_none(column)
    if arr is None:
        return None
    value = arr[i]
    if isinstance(value, float) and value != value:
        return None
    return value


def _nested(dataset: VulnDataset, column: str, i: int) -> Any:
    """Decode a nested JSON column at row i (None if absent or unparsable)."""
    raw = _cell(dataset, column, i)
    if raw is None:
        return None
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _iso(value: Any) -> str | None:
    """ISO 8601 UTC string for any timestamp form, or None."""
    try:
        dt = to_utc_datetime(value)
    except (TypeError, ValueError):
        return None
    return dt.isoformat().replace("+00:00", "Z") if dt else None


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if f != f else round(f, 5)


def _vector_metrics(vector: str | None) -> dict[str, str]:
    """Split a CVSS vector into {metric: value}, e.g. {'AV': 'N', 'AC': 'L'}."""
    if not vector:
        return {}
    parts = [p.split(":", 1) for p in vector.split("/") if ":" in p]
    return {k: v for k, v in parts if k != "CVSS"}


def _title(cve_id: str, kev_name: Any, description: Any) -> str:
    """A one-line title: KEV's name if listed, else the description's first sentence."""
    if isinstance(kev_name, str) and kev_name.strip():
        return kev_name.strip()
    if not isinstance(description, str) or not description.strip():
        return cve_id
    first = re.split(r"(?<=[.!?])\s", description.strip(), maxsplit=1)[0]
    if len(first) <= _TITLE_LENGTH:
        return first.rstrip(".")
    return first[: _TITLE_LENGTH - 1].rsplit(" ", 1)[0] + "…"


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _product(
    kev_vendor: Any, kev_product: Any, cpes: list[dict[str, Any]]
) -> tuple[str | None, str | None, list[str]]:
    """Vendor, product and fix versions, from KEV names or the first CPE.

    Fix versions come only from CPEs of the chosen vendor (and product, when a
    CPE product matches), so a downstream vendor's firmware fix is not shown
    as the fix for the library itself.
    """
    vendor = kev_vendor if isinstance(kev_vendor, str) and kev_vendor else None
    product = kev_product if isinstance(kev_product, str) and kev_product else None
    if vendor is None and cpes:
        vendor, product = cpes[0].get("vendor"), cpes[0].get("product")
    same_vendor = [
        c for c in cpes if vendor and _norm(str(c.get("vendor", ""))) == _norm(vendor)
    ]
    if product:
        p = _norm(product)
        same_product = [
            c
            for c in same_vendor
            if _norm(str(c.get("product", ""))) in p
            or p in _norm(str(c.get("product", "")))
        ]
        same_vendor = same_product or same_vendor
    fixes: list[str] = []
    for c in same_vendor:
        fix = c.get("fix_version")
        if isinstance(fix, str) and fix and fix not in fixes:
            fixes.append(fix)
    return vendor, product, fixes[:6]


# ============================================================================
# SSVC evidence heuristics
# ============================================================================


def suggest_exploitation(
    is_kev: bool, kev_added: str | None, exploit_refs: list[str], maturity: str | None
) -> dict[str, Any]:
    """SSVC Exploitation: active, poc or none."""
    if is_kev:
        when = f" on {kev_added[:10]}" if kev_added else ""
        return {"value": "active", "reason": f"Listed in CISA KEV{when}"}
    if maturity == "A":
        return {"value": "active", "reason": "CVSS v4 Exploit Maturity: Attacked"}
    if exploit_refs:
        n = len(exploit_refs)
        return {
            "value": "poc",
            "reason": f"{n} reference{'s' if n > 1 else ''} tagged Exploit by NVD",
        }
    if maturity == "P":
        return {"value": "poc", "reason": "CVSS v4 Exploit Maturity: POC"}
    return {"value": "none", "reason": "Not in KEV and no exploit references"}


def suggest_automatable(metrics: dict[str, str]) -> dict[str, Any] | None:
    """SSVC Automatable from CVSS: yes only with no barrier in AV, AC, PR, UI."""
    if not metrics:
        return None
    if metrics.get("AU") in ("Y", "N"):
        value = "yes" if metrics["AU"] == "Y" else "no"
        return {"value": value, "reason": "CVSS v4 Automatable supplemental metric"}
    barriers = []
    if metrics.get("AV") != "N":
        barriers.append(f"AV:{metrics.get('AV', '?')}")
    if metrics.get("AC") != "L":
        barriers.append(f"AC:{metrics.get('AC', '?')}")
    if metrics.get("AT") == "P":
        barriers.append("AT:P")
    if metrics.get("PR") != "N":
        barriers.append(f"PR:{metrics.get('PR', '?')}")
    if metrics.get("UI", "N") != "N":
        barriers.append(f"UI:{metrics['UI']}")
    if not barriers:
        return {
            "value": "yes",
            "reason": "AV:N, AC:L, PR:N, UI:N: no barrier to scripting at scale",
        }
    return {"value": "no", "reason": "Barrier to automation: " + ", ".join(barriers)}


def suggest_technical_impact(metrics: dict[str, str]) -> dict[str, Any] | None:
    """SSVC Technical Impact from CVSS: total needs high C and high I."""
    if not metrics:
        return None
    c = metrics.get("C", metrics.get("VC"))
    i = metrics.get("I", metrics.get("VI"))
    a = metrics.get("A", metrics.get("VA"))
    if c is None or i is None:
        return None
    if c == "H" and i == "H":
        return {"value": "total", "reason": "High confidentiality and integrity impact"}
    return {
        "value": "partial",
        "reason": f"Impact C:{c} I:{i} A:{a or '?'}: not full control",
    }


# ============================================================================
# Loader
# ============================================================================


@Registry.register_loader("triage")
class TriageLoader(BaseLoader):
    """Write a ranked SSVC triage queue as JSON for VulnTriage.

    Rows are filtered (NVD status, publication window, KEV-only, minimum EPSS),
    ranked (KEV first, then EPSS, then CVSS) and capped at ``limit``.

    Example:
        >>> from vulnflow.helpers import CacheBuilder
        >>> ds = CacheBuilder(recent_only=True).build()
        >>> TriageLoader("queue.json", min_epss=0.01).load(ds)
    """

    format: str = "json"
    name: str = "TriageLoader"

    def __init__(
        self,
        path: str | Path,
        *,
        limit: int | None = 500,
        start: str | datetime | None = None,
        end: str | datetime | None = None,
        kev_only: bool = False,
        min_epss: float | None = None,
        min_cvss: float | None = None,
        cve_ids: list[str] | None = None,
        include_awaiting: bool = True,
        include_deferred: bool = True,
    ) -> None:
        """Initialize loader.

        Args:
            path: Output JSON path
            limit: Maximum CVEs in the queue after ranking (None for all)
            start: Inclusive lower bound on nvd_published_at
            end: Exclusive upper bound on nvd_published_at
            kev_only: Keep only CVEs in CISA KEV
            min_epss: Keep only CVEs with EPSS probability at or above this
            min_cvss: Keep only CVEs with CVSS base score at or above this
            cve_ids: Keep only these CVE IDs (applied before other filters)
            include_awaiting: Include CVEs NVD has not analyzed yet
            include_deferred: Include CVEs NVD has deferred
        """
        super().__init__()
        if limit is not None and limit < 1:
            raise LoaderError(f"limit must be positive or None, got {limit}")
        self.path = Path(path)
        self.limit = limit
        self.start = start
        self.end = end
        self.kev_only = kev_only
        self.min_epss = min_epss
        self.min_cvss = min_cvss
        self.cve_ids = set(cve_ids) if cve_ids else None
        self.include_awaiting = include_awaiting
        self.include_deferred = include_deferred

    @property
    def label(self) -> str:
        """Short display name for progress/provenance."""
        return f"TriageLoader({self.path.name})"

    # ------------------------------------------------------------------
    def _mask(self, dataset: VulnDataset) -> np.ndarray:
        n = len(dataset)
        mask: np.ndarray = _build_status_mask(
            dataset,
            awaiting=self.include_awaiting,
            deferred=self.include_deferred,
        ) & _build_date_mask(dataset, start=self.start, end=self.end)
        if self.cve_ids is not None:
            ids = dataset.get_column_or_none("cve_id")
            if ids is None:
                return np.zeros(n, dtype=bool)
            mask &= np.array([x in self.cve_ids for x in ids], dtype=bool)
        if self.kev_only:
            kev = dataset.get_column_or_none("is_kev")
            mask &= (
                np.array([bool(x) for x in kev], dtype=bool)
                if kev is not None
                else np.zeros(n, dtype=bool)
            )
        if self.min_epss is not None:
            mask &= self._at_least(dataset, "epss_probability", self.min_epss)
        if self.min_cvss is not None:
            mask &= self._at_least(dataset, "cvss_base_score", self.min_cvss)
        return mask

    @staticmethod
    def _at_least(dataset: VulnDataset, column: str, floor: float) -> np.ndarray:
        arr = dataset.get_column_or_none(column)
        if arr is None:
            return np.zeros(len(dataset), dtype=bool)
        return np.array(
            [(_float(x) or 0.0) >= floor for x in arr],
            dtype=bool,
        )

    # ------------------------------------------------------------------
    def _item(self, dataset: VulnDataset, i: int) -> dict[str, Any]:
        cve_id = str(_cell(dataset, "cve_id", i))
        description = _cell(dataset, "description", i)

        vector = _cell(dataset, "cvss_vector_string", i)
        if vector is None:
            scores = _nested(dataset, "cvss_scores", i) or []
            if scores:
                vector = scores[0].get("vector_string")
        metrics = _vector_metrics(vector)
        score = _float(_cell(dataset, "cvss_base_score", i))

        kev = _nested(dataset, "kev", i) or {}
        is_kev = bool(_cell(dataset, "is_kev", i))
        kev_added = _iso(_cell(dataset, "kev_date_added", i) or kev.get("date_added"))

        references = _nested(dataset, "references", i) or []
        exploit_refs = [
            r["url"] for r in references if _EXPLOIT_TAG in (r.get("tags") or [])
        ]
        cwes = [c["id"] for c in (_nested(dataset, "cwes", i) or []) if c.get("id")]
        cpes = _nested(dataset, "cpes", i) or []

        vendor, product, fixed_in = _product(
            _cell(dataset, "kev_vendor", i), _cell(dataset, "kev_product", i), cpes
        )

        evidence: dict[str, Any] = {
            "exploitation": suggest_exploitation(
                is_kev, kev_added, exploit_refs, metrics.get("E")
            )
        }
        automatable = suggest_automatable(metrics)
        if automatable:
            evidence["automatable"] = automatable
        impact = suggest_technical_impact(metrics)
        if impact:
            evidence["technical_impact"] = impact

        return {
            "id": cve_id,
            "title": _title(
                cve_id, _cell(dataset, "kev_vulnerability_name", i), description
            ),
            "description": description,
            "published": _iso(_cell(dataset, "nvd_published_at", i)),
            "modified": _iso(_cell(dataset, "nvd_modified_at", i)),
            "nvd_status": _cell(dataset, "nvd_status", i),
            "product": {
                "vendor": vendor,
                "name": product,
                "fixed_in": fixed_in,
                "cpe_count": len(cpes),
            },
            "cvss": (
                {
                    "version": _cell(dataset, "cvss_version", i),
                    "score": score,
                    "severity": _cell(dataset, "cvss_base_severity", i),
                    "vector": vector,
                    "metrics": metrics,
                    "source": _cell(dataset, "cvss_source", i),
                }
                if vector
                else None
            ),
            "cwe": cwes,
            "epss": (
                {
                    "probability": _float(_cell(dataset, "epss_probability", i)),
                    "percentile": _float(_cell(dataset, "epss_percentile", i)),
                    "date": _iso(_cell(dataset, "epss_date", i)),
                }
                if _cell(dataset, "epss_probability", i) is not None
                else None
            ),
            "kev": {
                "listed": is_kev,
                "date_added": kev_added,
                "due_date": _iso(
                    _cell(dataset, "kev_due_date", i) or kev.get("due_date")
                ),
                "ransomware": bool(_cell(dataset, "kev_ransomware", i)),
                "required_action": _cell(dataset, "kev_required_action", i),
            },
            "exploit_refs": exploit_refs,
            "references": [
                {"url": r.get("url"), "tags": r.get("tags") or []}
                for r in references[:_MAX_REFERENCES]
            ],
            "evidence": evidence,
        }

    @staticmethod
    def _rank_key(item: dict[str, Any]) -> tuple[int, float, float, str]:
        epss = (item["epss"] or {}).get("probability") or 0.0
        score = (item["cvss"] or {}).get("score") or 0.0
        return (0 if item["kev"]["listed"] else 1, -epss, -score, item["id"])

    # ------------------------------------------------------------------
    def build(self, dataset: VulnDataset) -> dict[str, Any]:
        """Build the export document without writing it."""
        if len(dataset) and dataset.get_column_or_none("cve_id") is None:
            raise LoaderError("TriageLoader needs a 'cve_id' column")
        mask = self._mask(dataset) if len(dataset) else np.zeros(0, dtype=bool)
        items = [self._item(dataset, int(i)) for i in np.flatnonzero(mask)]
        items.sort(key=self._rank_key)
        matched = len(items)
        if self.limit is not None:
            items = items[: self.limit]

        from vulnflow import __version__

        return {
            "schema": TRIAGE_SCHEMA,
            "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "generator": f"vulnflow {__version__}",
            "filters": {
                "start": _iso(self.start) if self.start else None,
                "end": _iso(self.end) if self.end else None,
                "kev_only": self.kev_only,
                "min_epss": self.min_epss,
                "min_cvss": self.min_cvss,
                "limit": self.limit,
            },
            "enrichment": {
                "kev": dataset.get_column_or_none("is_kev") is not None,
                "epss": dataset.get_column_or_none("epss_probability") is not None,
                "cvss_flattened": dataset.get_column_or_none("cvss_vector_string")
                is not None,
            },
            "matched": matched,
            "count": len(items),
            "items": items,
        }

    def load(self, dataset: VulnDataset) -> dict[str, Any]:
        """Write the triage queue JSON.

        Returns:
            Metadata with path, record_count (exported) and matched (before limit)
        """
        doc = self.build(dataset)
        self._create_parent_dirs(self.path)
        self._write_json_safely(self.path, doc)
        return {
            "path": str(self.path),
            "record_count": doc["count"],
            "matched": doc["matched"],
        }
