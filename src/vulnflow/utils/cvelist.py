"""
cvelistV5 Parsing Utilities - Map CVE JSON 5.x records to VulnFlow columns

Provides:
- parse_cvelist_cvss: CNA (authoritative) and ADP CVSS v2.0/v3.0/v3.1/v4.0 scores
- parse_cvelist_cwes: CWE IDs from CNA, then ADP, problemTypes
- parse_cvelist_cpes: Explicit CPE strings (affected[].cpes, cpeApplicability)
- parse_cvelist_references: CNA and ADP references, one per URL
- populate_cvelist_columns: Convert cvelistV5 records to columnar arrays
- fetch_releases: List cvelistV5 GitHub releases
- latest_full_url: Newest "all CVEs at midnight" zip URL
- latest_delta_url: Newest hourly delta zip URL for a UTC day
- snapshot_date: UTC day of the release asset named in an extractor label

Layer: Utils
Dependencies: schema, utils.nvd, utils.cvss, utils.cpe, utils.http, utils.json
Used by: CVEListExtractor, CacheBuilder
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from vulnflow.schema import CPEEntry, CVSSv3Entry, CVSSv4Entry, CWEEntry, ReferenceEntry
from vulnflow.utils import json
from vulnflow.utils.cpe import parse_cpe23
from vulnflow.utils.cvss import score_to_severity
from vulnflow.utils.http import HttpFetcher
from vulnflow.utils.nvd import parse_cpes, parse_cvss, to_utc_datetime

RELEASES_URL = "https://api.github.com/repos/CVEProject/cvelistV5/releases"
FULL_SUFFIX = "_all_CVEs_at_midnight.zip.zip"
DELTA_MARKER = "_delta_CVEs_at_"

# cvelistV5 metric key -> NVD metrics key, highest version first so that
# CVSSFlattenTransform's "first authoritative" pick is the newest CNA score
_CVSS_KEYS = (
    ("cvssV4_0", "cvssMetricV40"),
    ("cvssV3_1", "cvssMetricV31"),
    ("cvssV3_0", "cvssMetricV30"),
    ("cvssV2_0", "cvssMetricV2"),
)

# Validation result per record: (valid, error code, first error message)
Validation = tuple[bool, str | None, str | None]


def _containers(containers: dict[str, Any]) -> list[tuple[dict[str, Any], bool]]:
    """(container, is_cna) pairs: CNA first, then each ADP."""
    pairs: list[tuple[dict[str, Any], bool]] = []
    if isinstance(containers.get("cna"), dict):
        pairs.append((containers["cna"], True))
    pairs += [(a, False) for a in containers.get("adp") or [] if isinstance(a, dict)]
    return pairs


def _short_name(container: dict[str, Any]) -> str:
    return str((container.get("providerMetadata") or {}).get("shortName") or "")


def _v2_severity(score: float) -> str:
    """CVSS v2 ranges (no CRITICAL band)."""
    if score >= 7.0:
        return "HIGH"
    return "MEDIUM" if score >= 4.0 else "LOW"


def parse_cvelist_cvss(
    containers: dict[str, Any],
) -> list[CVSSv3Entry | CVSSv4Entry]:
    """Parse CVSS scores from CNA and ADP containers.

    Reuses utils.nvd.parse_cvss by reshaping each container's metrics into
    NVD form. CNA scores are authoritative (``type="Primary"``), ADP scores
    are not. ``source`` is the provider's ``shortName``, normalized
    (lowercased). Missing severities are computed from the score.

    Args:
        containers: The record's ``containers`` dict (``cna`` and ``adp``).

    Returns:
        CNA entries first (v4.0, v3.1, v3.0, v2.0), then each ADP's.
    """
    scores: list[CVSSv3Entry | CVSSv4Entry] = []
    for container, is_cna in _containers(containers):
        nvd_metrics: dict[str, list[dict[str, Any]]] = {}
        for metric in container.get("metrics") or []:
            for key, nvd_key in _CVSS_KEYS:
                data = metric.get(key) if isinstance(metric, dict) else None
                if not isinstance(data, dict):
                    continue
                score = float(data.get("baseScore") or 0.0)
                severity = data.get("baseSeverity") or (
                    _v2_severity(score)
                    if key == "cvssV2_0"
                    else score_to_severity(score)
                )
                nvd_metrics.setdefault(nvd_key, []).append(
                    {
                        "cvssData": {**data, "baseSeverity": severity},
                        "baseSeverity": severity,  # v2 location in NVD form
                        "source": _short_name(container) or "unknown",
                        "type": "Primary" if is_cna else "Secondary",
                    }
                )
        scores += parse_cvss(nvd_metrics)
    return scores


def parse_cvelist_cwes(containers: dict[str, Any]) -> list[CWEEntry]:
    """Parse CWE IDs from problemTypes: CNA first, then ADP, without duplicates."""
    seen: dict[str, CWEEntry] = {}
    for container, _ in _containers(containers):
        for problem in container.get("problemTypes") or []:
            for desc in problem.get("descriptions") or []:
                cwe_id = desc.get("cweId") if isinstance(desc, dict) else None
                if isinstance(cwe_id, str) and cwe_id.startswith("CWE-"):
                    seen.setdefault(cwe_id, CWEEntry(id=cwe_id))
    return list(seen.values())


def _parse_cpe(cpe_str: str) -> CPEEntry | None:
    """Parse a CPE 2.3 string or a CPE 2.2 URI (``cpe:/part:vendor:product:version``)."""
    if cpe_str.startswith("cpe:2.3:"):
        try:
            return parse_cpe23(cpe_str)
        except ValueError:
            return None
    if cpe_str.startswith("cpe:/"):
        parts = cpe_str.split(":")
        if len(parts) >= 4 and parts[2] and parts[3]:
            version = parts[4] if len(parts) > 4 and parts[4] else None
            return CPEEntry(
                vendor=parts[2], product=parts[3], version=version, raw=cpe_str
            )
    return None


def parse_cvelist_cpes(containers: dict[str, Any]) -> list[CPEEntry]:
    """Parse explicit CPE strings only (decision 2).

    Sources, in order: ``cpeApplicability`` (NVD configuration format, keeps
    ``versionEndExcluding`` as fix_version), then ``affected[].cpes``, for the
    CNA and then each ADP. Vendor and product names from ``affected`` are not
    CPE names and are not mapped. One entry per raw CPE string.
    """
    seen: dict[str, CPEEntry] = {}
    for container, _ in _containers(containers):
        for cpe in parse_cpes(container.get("cpeApplicability") or []):
            seen.setdefault(cpe.raw or "", cpe)
        for affected in container.get("affected") or []:
            for cpe_str in affected.get("cpes") or []:
                if isinstance(cpe_str, str) and cpe_str not in seen:
                    parsed = _parse_cpe(cpe_str)
                    if parsed is not None:
                        seen[cpe_str] = parsed
    return list(seen.values())


def parse_cvelist_references(containers: dict[str, Any]) -> list[ReferenceEntry]:
    """Parse CNA then ADP references; the first entry per URL wins.

    ``source`` is the provider's ``shortName``, lowercased.
    """
    seen: dict[str, ReferenceEntry] = {}
    for container, _ in _containers(containers):
        source = _short_name(container).lower() or None
        for ref in container.get("references") or []:
            url = ref.get("url") if isinstance(ref, dict) else None
            if isinstance(url, str) and url and url not in seen:
                tags = ref.get("tags")
                seen[url] = ReferenceEntry(
                    url=url, source=source, tags=tags if isinstance(tags, list) else []
                )
    return list(seen.values())


def _english(entries: list[dict[str, Any]] | None) -> str | None:
    """First value whose lang starts with "en" (en, en-US, ...)."""
    for entry in entries or []:
        if str(entry.get("lang", "")).lower().startswith("en"):
            value = entry.get("value")
            return value if isinstance(value, str) else None
    return None


def _ssvc(containers: dict[str, Any]) -> Any:
    """SSVC decision content from the first ADP that has one (CISA-ADP)."""
    for container, _ in _containers(containers):
        for metric in container.get("metrics") or []:
            other = metric.get("other") if isinstance(metric, dict) else None
            if isinstance(other, dict) and other.get("type") == "ssvc":
                return other.get("content")
    return None


def _metadata(record: dict[str, Any], validation: Validation | None) -> dict[str, Any]:
    """cvelistV5-only fields kept as JSON in the ``metadata`` column."""
    meta = record.get("cveMetadata") or {}
    containers = record.get("containers") or {}
    out: dict[str, Any] = {
        "source": "cvelistV5",
        "assigner": meta.get("assignerShortName"),
        "state": meta.get("state"),
        "date_reserved": meta.get("dateReserved"),
    }
    ssvc = _ssvc(containers)
    if ssvc is not None:
        out["ssvc"] = ssvc
    affected = [
        {k: a[k] for k in ("vendor", "product", "versions", "packageURL") if k in a}
        for a in (containers.get("cna") or {}).get("affected") or []
        if isinstance(a, dict)
    ]
    if affected:
        out["affected"] = affected
    if validation is not None and not validation[0]:
        out["validation_message"] = validation[2]
    return out


def _json_list(entries: list[Any]) -> str:
    return json.dumps([e.to_dict() for e in entries]) if entries else "[]"


def populate_cvelist_columns(
    records: list[dict[str, Any]],
    validations: list[Validation] | None = None,
) -> dict[str, np.ndarray[Any, Any]]:
    """Convert cvelistV5 records to columnar arrays (one VulnRecord field each).

    Dates and status reuse the ``nvd_*`` columns (decision 1):
    ``datePublished`` → ``nvd_published_at``, ``dateUpdated`` →
    ``nvd_modified_at``, ``state`` → ``nvd_status`` (``published`` /
    ``rejected``). Rejected records take their description from
    ``rejectedReasons``. Complex fields are JSON-encoded strings.

    Args:
        records: Parsed CVE JSON 5.x records.
        validations: Optional ``(valid, error, message)`` per record. When
            given, adds ``cve_valid`` (bool) and ``cve_validation_error``
            columns and puts the message in ``metadata.validation_message``.

    Returns:
        Dict of column name → numpy array. Empty dict for empty input.

    Raises:
        ValueError: If validations is given with a different length.
    """
    from vulnflow.utils.schema import get_record_columns

    n = len(records)
    if validations is not None and len(validations) != n:
        raise ValueError(f"{len(validations)} validations for {n} records")
    if n == 0:
        return {}

    arrays: dict[str, np.ndarray[Any, Any]] = {
        col: np.empty(n, dtype=object) for col in get_record_columns()
    }
    for i, record in enumerate(records):
        meta = record.get("cveMetadata") or {}
        containers = record.get("containers") or {}
        cna = containers.get("cna") or {}
        state = str(meta.get("state") or "").lower()

        arrays["id"][i] = str(uuid4())
        arrays["cve_id"][i] = meta.get("cveId")
        arrays["description"][i] = _english(
            cna.get("rejectedReasons")
            if state == "rejected"
            else cna.get("descriptions")
        )
        arrays["nvd_published_at"][i] = to_utc_datetime(meta.get("datePublished"))
        arrays["nvd_modified_at"][i] = to_utc_datetime(meta.get("dateUpdated"))
        arrays["nvd_status"][i] = state or None

        arrays["cvss_scores"][i] = _json_list(parse_cvelist_cvss(containers))
        arrays["cwes"][i] = _json_list(parse_cvelist_cwes(containers))
        arrays["cpes"][i] = _json_list(parse_cvelist_cpes(containers))
        arrays["references"][i] = _json_list(parse_cvelist_references(containers))

        arrays["epss_scores"][i] = "[]"
        arrays["kev"][i] = "null"
        arrays["exploits"][i] = "[]"
        arrays["capecs"][i] = "[]"
        arrays["stride_threats"][i] = "[]"
        arrays["tags"][i] = "{}"
        arrays["metadata"][i] = json.dumps(
            _metadata(record, validations[i] if validations else None)
        )

    if validations is not None:
        arrays["cve_valid"] = np.array([v[0] for v in validations], dtype=bool)
        arrays["cve_validation_error"] = np.array(
            [v[1] for v in validations], dtype=object
        )
    return arrays


# ============================================================================
# Release resolution (GitHub releases of CVEProject/cvelistV5)
# ============================================================================


def fetch_releases(timeout: float = 30.0, per_page: int = 30) -> list[dict[str, Any]]:
    """List the newest cvelistV5 GitHub releases (newest first).

    One unauthenticated GitHub API call (limit: 60 per hour). Each hourly
    release carries the day's midnight zip and a delta zip that is cumulative
    since midnight UTC, so the first page always holds today's assets.

    Raises:
        HttpError: On network or HTTP errors.
    """
    fetcher = HttpFetcher(cache_dir=Path(".cache/cvelist"), timeout=timeout)
    content, _ = fetcher.fetch_sync(
        f"{RELEASES_URL}?per_page={per_page}",
        headers={"Accept": "application/vnd.github+json"},
    )
    releases: list[dict[str, Any]] = json.loads(content)
    return releases


def _asset_urls(releases: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """(asset name, download URL) across releases, in the given order."""
    return [
        (str(a.get("name", "")), str(a.get("browser_download_url", "")))
        for r in releases
        for a in r.get("assets") or []
    ]


def latest_full_url(releases: list[dict[str, Any]]) -> str:
    """Download URL of the newest ``<date>_all_CVEs_at_midnight.zip.zip``.

    Raises:
        ValueError: If no release has a midnight zip.
    """
    fulls = [(n, u) for n, u in _asset_urls(releases) if n.endswith(FULL_SUFFIX)]
    if not fulls:
        raise ValueError("No cvelistV5 release has an all_CVEs_at_midnight zip")
    return max(fulls)[1]  # names start with the date, so max is the newest


def latest_delta_url(releases: list[dict[str, Any]], day: date) -> str | None:
    """URL of the day's newest hourly delta zip (cumulative since midnight UTC).

    Returns:
        URL of ``<day>_delta_CVEs_at_<HHMM>Z.zip`` with the latest HHMM, or
        None if the releases hold no delta for that day.
    """
    prefix = f"{day.isoformat()}{DELTA_MARKER}"
    deltas = [(n, u) for n, u in _asset_urls(releases) if n.startswith(prefix)]
    return max(deltas)[1] if deltas else None


_SNAPSHOT = re.compile(r"(\d{4}-\d{2}-\d{2})_(?:all_CVEs_at_midnight|delta_CVEs_at_)")


def snapshot_date(label: str) -> date | None:
    """UTC day of the release asset named in an extractor label, if any.

    Example:
        >>> snapshot_date("CVEListExtractor(2026-11-13_delta_CVEs_at_1700Z.zip)")
        datetime.date(2026, 11, 13)
    """
    match = _SNAPSHOT.search(label)
    return date.fromisoformat(match.group(1)) if match else None
