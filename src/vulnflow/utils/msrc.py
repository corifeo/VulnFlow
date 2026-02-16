"""MSRC Parsing Utilities - Parse MSRC CVRF v3.0 JSON structures.

Provides:
- to_msrc_month_id: Convert YYYY-MM to YYYY-Mon format
- resolve_month_ids: Filter available months by year/start/end params
- parse_msrc_cvss: Parse CVSSScoreSets to CVSSv3Entry list
- parse_msrc_cwes: Parse CWE object to CWEEntry list
- parse_msrc_references: Parse Remediations to ReferenceEntry list
- build_msrc_metadata: Build metadata dict from Threats/Remediations
- populate_msrc_columns: Convert CVRF vulnerabilities to columnar arrays

Layer: Utils
Dependencies: schema, utils.json, utils.schema, utils.cvss
Used by: MSRCExtractor
"""

from __future__ import annotations

import calendar
import logging
import re
from typing import Any
from uuid import uuid4

import numpy as np

from vulnflow.schema import CVSSv3Entry, CWEEntry, ReferenceEntry
from vulnflow.utils.cvss import score_to_severity

logger = logging.getLogger(__name__)

# Month abbreviation mapping (1-indexed)
_MONTH_ABBREVS = {i: calendar.month_abbr[i] for i in range(1, 13)}
_ABBREV_TO_NUM = {v: k for k, v in _MONTH_ABBREVS.items()}

# Regex to extract CVSS version from vector string
_CVSS_VERSION_RE = re.compile(r"^CVSS:(\d+\.\d+)/")


def to_msrc_month_id(iso_month: str) -> str:
    """Convert YYYY-MM to MSRC month ID format YYYY-Mon.

    Args:
        iso_month: Month in YYYY-MM format (e.g., "2025-04")

    Returns:
        MSRC month ID (e.g., "2025-Apr")

    Raises:
        ValueError: If format is invalid or month is out of range
    """
    try:
        year_str, month_str = iso_month.split("-")
        month_num = int(month_str)
    except (ValueError, AttributeError):
        msg = f"Invalid month format: {iso_month!r}. Expected YYYY-MM."
        raise ValueError(msg) from None

    if month_num < 1 or month_num > 12:
        msg = f"Invalid month format: {iso_month!r}. Month must be 01-12."
        raise ValueError(msg)

    return f"{year_str}-{_MONTH_ABBREVS[month_num]}"


def _msrc_id_to_sort_key(msrc_id: str) -> tuple[int, int]:
    """Convert MSRC month ID to sortable (year, month) tuple."""
    year_str, abbrev = msrc_id.split("-", maxsplit=1)
    return int(year_str), _ABBREV_TO_NUM.get(abbrev, 0)


def resolve_month_ids(
    *,
    year: int | list[int] | None,
    start: str | None,
    end: str | None,
    available: list[str],
) -> list[str]:
    """Resolve constructor params to list of MSRC month IDs.

    Args:
        year: Single year or list of years to include
        start: Start month in YYYY-MM format (inclusive)
        end: End month in YYYY-MM format (inclusive). Requires start.
        available: List of available MSRC month IDs from /updates

    Returns:
        Filtered and sorted list of MSRC month IDs

    Raises:
        ValueError: On conflicting params or no matching months
    """
    if year is not None and start is not None:
        msg = "Cannot combine 'year' and 'start'/'end' parameters"
        raise ValueError(msg)

    if end is not None and start is None:
        msg = "'end' requires 'start' parameter"
        raise ValueError(msg)

    # No args → current month (latest available)
    if year is None and start is None:
        if not available:
            msg = "No matching months found in available releases"
            raise ValueError(msg)
        # Return the latest available month
        sorted_avail = sorted(available, key=_msrc_id_to_sort_key)
        return [sorted_avail[-1]]

    # Year filter
    if year is not None:
        years = [year] if isinstance(year, int) else year
        year_strs = {str(y) for y in years}
        result = [m for m in available if m.split("-")[0] in year_strs]
        result.sort(key=_msrc_id_to_sort_key)
        if not result:
            msg = f"No matching months found for year(s) {years}"
            raise ValueError(msg)
        return result

    # Start/end range
    assert start is not None
    start_id = to_msrc_month_id(start)
    start_key = _msrc_id_to_sort_key(start_id)

    if end is not None:
        end_id = to_msrc_month_id(end)
        end_key = _msrc_id_to_sort_key(end_id)
    else:
        # start only → single month
        end_key = start_key

    result = [m for m in available if start_key <= _msrc_id_to_sort_key(m) <= end_key]
    result.sort(key=_msrc_id_to_sort_key)

    if not result:
        msg = f"No matching months found for range {start} to {end or start}"
        raise ValueError(msg)
    return result


def parse_msrc_cvss(
    score_sets: list[dict[str, Any]],
) -> list[CVSSv3Entry]:
    """Parse CVSSScoreSets from MSRC CVRF format.

    Deduplicates by vector string, merging product_ids from duplicate vectors.

    Args:
        score_sets: List of MSRC CVSSScoreSet dicts

    Returns:
        List of CVSSv3Entry instances (unique by vector)
    """
    # Deduplicate by vector string, collecting product_ids
    seen: dict[str, dict[str, Any]] = {}

    for ss in score_sets:
        vector = ss.get("Vector", "")
        if not vector:
            continue

        product_ids = ss.get("ProductID", [])

        if vector in seen:
            # Merge product_ids
            existing_pids = seen[vector]["product_ids"]
            for pid in product_ids:
                if pid not in existing_pids:
                    existing_pids.append(pid)
        else:
            seen[vector] = {
                "base_score": ss.get("BaseScore", 0.0),
                "vector": vector,
                "product_ids": list(product_ids),
            }

    entries: list[CVSSv3Entry] = []
    for data in seen.values():
        vector = data["vector"]
        base_score = data["base_score"]

        # Extract version from vector string
        m = _CVSS_VERSION_RE.match(vector)
        version = m.group(1) if m else "3.1"

        # Parse component metrics from vector
        components = _parse_cvss_vector_components(vector)

        entries.append(
            CVSSv3Entry(
                version=version,
                vector_string=vector,
                base_score=base_score,
                base_severity=score_to_severity(base_score),
                source="microsoft",
                is_authoritative=True,
                attack_vector=components.get("AV"),
                attack_complexity=components.get("AC"),
                privileges_required=components.get("PR"),
                user_interaction=components.get("UI"),
                scope=components.get("S"),
                confidentiality_impact=components.get("C"),
                integrity_impact=components.get("I"),
                availability_impact=components.get("A"),
                metadata={"product_ids": data["product_ids"]},
            )
        )

    return entries


_CVSS_METRIC_MAP: dict[str, dict[str, str]] = {
    "AV": {"N": "NETWORK", "A": "ADJACENT_NETWORK", "L": "LOCAL", "P": "PHYSICAL"},
    "AC": {"L": "LOW", "H": "HIGH"},
    "PR": {"N": "NONE", "L": "LOW", "H": "HIGH"},
    "UI": {"N": "NONE", "R": "REQUIRED"},
    "S": {"U": "UNCHANGED", "C": "CHANGED"},
    "C": {"N": "NONE", "L": "LOW", "H": "HIGH"},
    "I": {"N": "NONE", "L": "LOW", "H": "HIGH"},
    "A": {"N": "NONE", "L": "LOW", "H": "HIGH"},
}


def _parse_cvss_vector_components(vector: str) -> dict[str, str]:
    """Extract base metric components from CVSS vector string."""
    components: dict[str, str] = {}
    # Remove version prefix
    parts = vector.split("/")
    for part in parts:
        if ":" not in part:
            continue
        key, val = part.split(":", 1)
        if key in _CVSS_METRIC_MAP:
            expanded = _CVSS_METRIC_MAP[key].get(val, val)
            components[key] = expanded
    return components


def parse_msrc_cwes(
    cwe_data: dict[str, Any] | list[dict[str, Any]] | None,
) -> list[CWEEntry]:
    """Parse CWE from MSRC CVRF format.

    MSRC may use a single CWE object or a list of CWE objects per vulnerability.

    Args:
        cwe_data: MSRC CWE dict or list of dicts with "ID" and "Value" keys

    Returns:
        List of CWEEntry instances
    """
    if not cwe_data:
        return []

    # Normalize to list
    items = cwe_data if isinstance(cwe_data, list) else [cwe_data]

    entries: list[CWEEntry] = []
    for item in items:
        cwe_id = item.get("ID", "")
        if cwe_id:
            entries.append(CWEEntry(id=cwe_id, name=item.get("Value")))
    return entries


def parse_msrc_references(
    remediations: list[dict[str, Any]],
) -> list[ReferenceEntry]:
    """Parse Remediations to ReferenceEntry list.

    Deduplicates by URL.

    Args:
        remediations: MSRC Remediations array

    Returns:
        List of ReferenceEntry (unique by URL)
    """
    seen_urls: set[str] = set()
    entries: list[ReferenceEntry] = []

    for rem in remediations:
        url = rem.get("URL", "")
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)

        entries.append(
            ReferenceEntry(
                url=url,
                source="msrc",
                tags=["patch", "kb-article"],
            )
        )

    return entries


def build_msrc_metadata(
    vuln: dict[str, Any],
    month_id: str,
) -> dict[str, Any]:
    """Build MSRC metadata dict from vulnerability data.

    Extracts exploitability, impact, severity, and remediation info
    from MSRC Threats and Remediations arrays.

    Args:
        vuln: MSRC vulnerability dict
        month_id: MSRC month ID (e.g., "2025-Apr")

    Returns:
        Metadata dict for metadata["msrc"]
    """
    threats = vuln.get("Threats", [])

    # Parse threats by type
    exploitability = ""
    exploited = False
    publicly_disclosed = False
    impact = ""
    severity_by_product_id: dict[str, str] = {}

    for threat in threats:
        threat_type = threat.get("Type")
        desc_value = threat.get("Description", {}).get("Value", "")
        product_ids = threat.get("ProductID", [])

        if threat_type == 0:
            # Impact type
            impact = desc_value
        elif threat_type == 1:
            # Exploitability assessment
            exploitability = desc_value
            desc_lower = desc_value.lower()
            if "detected" in desc_lower or "exploited" in desc_lower:
                exploited = True
            if "publicly disclosed" in desc_lower or "public" in desc_lower:
                publicly_disclosed = True
        elif threat_type == 3:
            # Severity per product
            for pid in product_ids:
                severity_by_product_id[pid] = desc_value

    # Collect product_ids from ProductStatuses
    product_ids_list: list[str] = []
    for ps in vuln.get("ProductStatuses", []):
        for pid in ps.get("ProductID", []):
            if pid not in product_ids_list:
                product_ids_list.append(pid)

    # Parse remediations
    rem_list: list[dict[str, Any]] = []
    for rem in vuln.get("Remediations", []):
        rem_entry: dict[str, Any] = {
            "type": rem.get("Type"),
            "description": rem.get("Description", {}).get("Value", ""),
            "url": rem.get("URL", ""),
            "product_ids": rem.get("ProductID", []),
            "sub_type": rem.get("SubType", ""),
        }
        restart = rem.get("RestartRequired")
        if restart:
            rem_entry["restart_required"] = restart.get("Value", "")
        fixed_build = rem.get("FixedBuild")
        if fixed_build:
            rem_entry["fixed_build"] = fixed_build
        supercedence = rem.get("Supercedence")
        if supercedence:
            rem_entry["supercedence"] = supercedence
        rem_list.append(rem_entry)

    # Parse acknowledgments
    ack_list: list[dict[str, Any]] = []
    for ack in vuln.get("Acknowledgments", []):
        names = [n.get("Value", "") for n in ack.get("Name", []) if n.get("Value")]
        if names:
            ack_entry: dict[str, Any] = {"names": names}
            ack_url = ack.get("URL")
            if ack_url:
                ack_entry["url"] = ack_url
            ack_list.append(ack_entry)

    return {
        "month_id": month_id,
        "title": vuln.get("Title", {}).get("Value", ""),
        "exploited": exploited,
        "publicly_disclosed": publicly_disclosed,
        "exploitability": exploitability,
        "impact": impact,
        "product_ids": product_ids_list,
        "severity_by_product_id": severity_by_product_id,
        "remediations": rem_list,
        "acknowledgments": ack_list,
    }


def populate_msrc_columns(
    vulnerabilities: list[dict[str, Any]],
    month_id: str,
) -> dict[str, np.ndarray[Any, Any]]:
    """Convert MSRC CVRF vulnerability dicts to columnar arrays.

    Single-pass conversion from MSRC JSON structures to pre-allocated
    numpy arrays. Complex fields are serialized to JSON strings.

    Args:
        vulnerabilities: List of MSRC vulnerability dicts from CVRF document
        month_id: MSRC month ID (e.g., "2025-Apr")

    Returns:
        Dict mapping column names to numpy object arrays. Empty dict for empty input.
    """
    from vulnflow.utils import json as json_util
    from vulnflow.utils.schema import get_record_columns

    n = len(vulnerabilities)
    if n == 0:
        return {}

    record_cols = get_record_columns()
    arrays: dict[str, np.ndarray[Any, Any]] = {
        col: np.empty(n, dtype=object) for col in record_cols
    }

    for i, vuln in enumerate(vulnerabilities):
        # Identity
        arrays["id"][i] = str(uuid4())
        arrays["cve_id"][i] = vuln.get("CVE")

        # Description (Notes where Title="Description")
        description = None
        for note in vuln.get("Notes", []):
            if note.get("Title") == "Description":
                description = note.get("Value")
                break
        arrays["description"][i] = description

        # NVD temporal fields (empty for MSRC)
        arrays["nvd_published_at"][i] = None
        arrays["nvd_modified_at"][i] = None
        arrays["nvd_status"][i] = None

        # CVSS scores
        cvss_entries = parse_msrc_cvss(vuln.get("CVSSScoreSets", []))
        cvss_dicts = [e.to_dict() for e in cvss_entries]
        arrays["cvss_scores"][i] = json_util.dumps(cvss_dicts) if cvss_dicts else "[]"

        # CWEs
        cwe_entries = parse_msrc_cwes(vuln.get("CWE"))
        cwe_dicts = [e.to_dict() for e in cwe_entries]
        arrays["cwes"][i] = json_util.dumps(cwe_dicts) if cwe_dicts else "[]"

        # CPEs (deferred to future MSRCProductEnricher)
        arrays["cpes"][i] = "[]"

        # References
        ref_entries = parse_msrc_references(vuln.get("Remediations", []))
        ref_dicts = [e.to_dict() for e in ref_entries]
        arrays["references"][i] = json_util.dumps(ref_dicts) if ref_dicts else "[]"

        # Empty fields (not in MSRC)
        arrays["epss_scores"][i] = "[]"
        arrays["kev"][i] = "null"
        arrays["exploits"][i] = "[]"
        arrays["capecs"][i] = "[]"
        arrays["stride_threats"][i] = "[]"
        arrays["tags"][i] = "{}"

        # MSRC metadata
        msrc_meta = build_msrc_metadata(vuln, month_id)
        arrays["metadata"][i] = json_util.dumps({"msrc": msrc_meta})

    return arrays
