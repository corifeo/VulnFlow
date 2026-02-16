"""
NVD Parsing Utilities - Parse NVD 2.0 JSON structures

Provides:
- parse_timestamp: ISO timestamp string to datetime
- extract_nvd_status: Normalize NVD vulnStatus field
- parse_cvss: Parse CVSS v2/v3.0/v3.1/v4.0 metrics
- parse_cwes: Parse CWE weaknesses
- parse_cpes: Parse CPE configurations
- parse_references: Parse reference entries
- populate_columns: Convert NVD vulnerability dicts to columnar arrays

Layer: Utils
Dependencies: schema, utils.cvss, utils.cpe, utils.json, utils.schema
Used by: NVDExtractor, NVDAPIExtractor
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

import numpy as np

from vulnflow.schema import CPEEntry, CVSSv3Entry, CVSSv4Entry, CWEEntry, ReferenceEntry
from vulnflow.utils.cpe import parse_cpe23
from vulnflow.utils.cvss import extract_cvss_metadata, score_to_severity


def parse_timestamp(ts: str | datetime | None) -> datetime | None:
    """Parse ISO timestamp string or datetime to datetime.

    Accepts ISO 8601 strings (with or without Z suffix), datetime
    objects (pass-through), or None.
    """
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts
    if not ts:
        return None
    # Handle Z suffix
    ts = ts.replace("Z", "+00:00")
    return datetime.fromisoformat(ts)


def extract_nvd_status(cve_data: dict[str, Any]) -> str:
    """Extract and normalize NVD status from vulnStatus field.

    NVD statuses (https://nvd.nist.gov/vuln/vulnerability-status):
    Analyzed, Awaiting Analysis, Undergoing Analysis, Received,
    Modified, Deferred, Rejected.

    Pre-2017 CVEs lack vulnStatus. For these, status is inferred:
    - Description contains ``** REJECTED **`` → ``"rejected"``
    - Otherwise → ``"analyzed"``

    Args:
        cve_data: The CVE data dict

    Returns:
        Normalized status: analyzed, awaiting, modified, rejected, deferred
    """
    vuln_status = cve_data.get("vulnStatus", "")
    if not vuln_status:
        # Pre-2017 CVEs lack vulnStatus. Check description for rejection.
        desc = (cve_data.get("descriptions") or [{}])[0].get("value", "")
        if "** REJECTED **" in desc:
            return "rejected"
        return "analyzed"

    status_lower = vuln_status.lower()

    # Map NVD status values to normalized form
    if "analyzed" in status_lower:
        return "analyzed"
    elif "awaiting" in status_lower:
        return "awaiting"
    elif "received" in status_lower:
        return "awaiting"  # Treat "Received" as awaiting analysis
    elif "undergoing" in status_lower:
        return "awaiting"  # Treat "Undergoing Analysis" as awaiting
    elif "modified" in status_lower:
        return "modified"
    elif "rejected" in status_lower:
        return "rejected"
    elif "deferred" in status_lower:
        return "deferred"
    else:
        return "analyzed"  # Safe fallback for unrecognized values


def parse_cvss(
    metrics: dict[str, list[dict[str, Any]]],
) -> list[CVSSv3Entry | CVSSv4Entry]:
    """Parse CVSS metrics from NVD format.

    Handles version-specific field locations and component fields:
    - v4.0: baseSeverity from cvssData (fallback: computed from score),
      reads vuln/sub CIA impacts and supplemental metrics
    - v3.1/v3.0: baseSeverity in cvssData, reads component fields
    - v2.0: baseSeverity at metric level (not in cvssData),
      maps accessVector→attack_vector, accessComplexity→attack_complexity

    Args:
        metrics: NVD metrics dict (e.g., cve_data["metrics"])

    Returns:
        List of CVSSv3Entry or CVSSv4Entry dataclass instances
    """
    cvss_scores: list[CVSSv3Entry | CVSSv4Entry] = []

    # CVSS 4.0 - prefer NVD baseSeverity, fall back to computed
    for metric in metrics.get("cvssMetricV40", []):
        cvss_data = metric.get("cvssData", {})
        metadata = extract_cvss_metadata(metric)
        base_score = cvss_data.get("baseScore", 0.0)

        cvss_scores.append(
            CVSSv4Entry(
                version=cvss_data.get("version", "4.0"),
                vector_string=cvss_data.get("vectorString", ""),
                base_score=base_score,
                base_severity=cvss_data.get("baseSeverity")
                or score_to_severity(base_score),
                # v4.0 base metrics
                attack_vector=cvss_data.get("attackVector"),
                attack_complexity=cvss_data.get("attackComplexity"),
                attack_requirements=cvss_data.get("attackRequirements"),
                privileges_required=cvss_data.get("privilegesRequired"),
                user_interaction=cvss_data.get("userInteraction"),
                # v4.0 vulnerable system impacts
                vuln_confidentiality_impact=cvss_data.get("vulnConfidentialityImpact"),
                vuln_integrity_impact=cvss_data.get("vulnIntegrityImpact"),
                vuln_availability_impact=cvss_data.get("vulnAvailabilityImpact"),
                # v4.0 subsequent system impacts
                sub_confidentiality_impact=cvss_data.get("subConfidentialityImpact"),
                sub_integrity_impact=cvss_data.get("subIntegrityImpact"),
                sub_availability_impact=cvss_data.get("subAvailabilityImpact"),
                # v4.0 supplemental metrics (NB: inconsistent casing in NVD)
                exploit_maturity=cvss_data.get("exploitMaturity"),
                automatable=cvss_data.get("Automatable"),
                recovery=cvss_data.get("Recovery"),
                safety=cvss_data.get("Safety"),
                provider_urgency=cvss_data.get("providerUrgency"),
                vulnerability_response_effort=cvss_data.get(
                    "vulnerabilityResponseEffort"
                ),
                value_density=cvss_data.get("valueDensity"),
                **metadata,
            )
        )

    # CVSS 3.1 - baseSeverity in cvssData
    for metric in metrics.get("cvssMetricV31", []):
        cvss_data = metric.get("cvssData", {})
        metadata = extract_cvss_metadata(metric)

        cvss_scores.append(
            CVSSv3Entry(
                version=cvss_data.get("version", "3.1"),
                vector_string=cvss_data.get("vectorString", ""),
                base_score=cvss_data.get("baseScore", 0.0),
                base_severity=cvss_data.get("baseSeverity", "NONE"),
                # v3.x component fields
                attack_vector=cvss_data.get("attackVector"),
                attack_complexity=cvss_data.get("attackComplexity"),
                privileges_required=cvss_data.get("privilegesRequired"),
                user_interaction=cvss_data.get("userInteraction"),
                scope=cvss_data.get("scope"),
                confidentiality_impact=cvss_data.get("confidentialityImpact"),
                integrity_impact=cvss_data.get("integrityImpact"),
                availability_impact=cvss_data.get("availabilityImpact"),
                **metadata,
            )
        )

    # CVSS 3.0 - same structure as 3.1
    for metric in metrics.get("cvssMetricV30", []):
        cvss_data = metric.get("cvssData", {})
        metadata = extract_cvss_metadata(metric)

        cvss_scores.append(
            CVSSv3Entry(
                version=cvss_data.get("version", "3.0"),
                vector_string=cvss_data.get("vectorString", ""),
                base_score=cvss_data.get("baseScore", 0.0),
                base_severity=cvss_data.get("baseSeverity", "NONE"),
                # v3.x component fields
                attack_vector=cvss_data.get("attackVector"),
                attack_complexity=cvss_data.get("attackComplexity"),
                privileges_required=cvss_data.get("privilegesRequired"),
                user_interaction=cvss_data.get("userInteraction"),
                scope=cvss_data.get("scope"),
                confidentiality_impact=cvss_data.get("confidentialityImpact"),
                integrity_impact=cvss_data.get("integrityImpact"),
                availability_impact=cvss_data.get("availabilityImpact"),
                **metadata,
            )
        )

    # CVSS 2.0 - baseSeverity at metric level (not in cvssData)
    # NVD v2 uses accessVector/accessComplexity (not attackVector/attackComplexity)
    for metric in metrics.get("cvssMetricV2", []):
        cvss_data = metric.get("cvssData", {})
        metadata = extract_cvss_metadata(metric)

        cvss_scores.append(
            CVSSv3Entry(
                version=cvss_data.get("version", "2.0"),
                vector_string=cvss_data.get("vectorString", ""),
                base_score=cvss_data.get("baseScore", 0.0),
                base_severity=metric.get("baseSeverity", "NONE"),  # Different location!
                # v2 field name mapping: accessVector → attack_vector, etc.
                attack_vector=cvss_data.get("accessVector"),
                attack_complexity=cvss_data.get("accessComplexity"),
                confidentiality_impact=cvss_data.get("confidentialityImpact"),
                integrity_impact=cvss_data.get("integrityImpact"),
                availability_impact=cvss_data.get("availabilityImpact"),
                **metadata,
            )
        )

    return cvss_scores


def parse_cwes(weaknesses: list[dict[str, object]]) -> list[CWEEntry]:
    """Parse CWE weaknesses from NVD format."""
    cwes: list[CWEEntry] = []

    for weakness in weaknesses:
        descriptions = weakness.get("description", [])
        if not descriptions or not isinstance(descriptions, list):
            continue

        for desc in descriptions:
            if not isinstance(desc, dict):
                continue
            cwe_id = desc.get("value", "")
            if cwe_id and isinstance(cwe_id, str) and cwe_id.startswith("CWE-"):
                cwes.append(
                    CWEEntry(
                        id=cwe_id,
                        name=None,
                        category_id=None,
                    )
                )

    return cwes


def parse_cpes(configurations: list[dict[str, object]]) -> list[CPEEntry]:
    """Parse CPE configurations from NVD format."""
    cpes: list[CPEEntry] = []

    for config in configurations:
        nodes = config.get("nodes", [])
        if not nodes or not isinstance(nodes, list):
            continue

        for node in nodes:
            if not isinstance(node, dict):
                continue
            cpe_matches = node.get("cpeMatch", [])
            if not cpe_matches or not isinstance(cpe_matches, list):
                continue

            for cpe_match in cpe_matches:
                if not isinstance(cpe_match, dict):
                    continue
                cpe_str = cpe_match.get("criteria", "")
                if (
                    cpe_str
                    and isinstance(cpe_str, str)
                    and cpe_str.startswith("cpe:2.3")
                ):
                    try:
                        cpe = parse_cpe23(cpe_str)
                        # Extract versionEndExcluding as fix_version
                        fix_version = cpe_match.get("versionEndExcluding")
                        if fix_version and isinstance(fix_version, str):
                            cpe.fix_version = fix_version
                        cpes.append(cpe)
                    except (ValueError, AttributeError):
                        # Skip invalid CPE strings or malformed data
                        pass

    return cpes


def parse_references(references: list[dict[str, object]]) -> list[ReferenceEntry]:
    """Parse references from NVD format."""
    ref_entries: list[ReferenceEntry] = []

    for ref in references:
        url = ref.get("url", "")
        if url and isinstance(url, str):
            source = ref.get("source")
            tags = ref.get("tags", [])
            ref_entries.append(
                ReferenceEntry(
                    url=url,
                    source=source if isinstance(source, str) else None,
                    tags=tags if isinstance(tags, list) else [],
                )
            )

    return ref_entries


def populate_columns(
    vulnerabilities: list[dict[str, Any]],
) -> dict[str, np.ndarray[Any, Any]]:
    """Convert NVD vulnerability dicts to columnar arrays.

    Single-pass conversion from NVD JSON structures to pre-allocated
    numpy arrays. Uses the parse functions defined in this module
    for CVSS, CWE, CPE, reference, timestamp, and status extraction.

    Complex fields (lists of dataclasses) are serialized to JSON strings.
    Fields not present in NVD data (epss, kev, exploits, capecs,
    stride_threats) are initialized to empty defaults.

    Args:
        vulnerabilities: List of NVD vulnerability dicts, each with a
            ``"cve"`` key containing the CVE data.

    Returns:
        Dict mapping column names to numpy object arrays, one entry per
        VulnRecord field. Returns empty dict for empty input.
    """
    from vulnflow.utils import json
    from vulnflow.utils.schema import get_record_columns

    n = len(vulnerabilities)
    if n == 0:
        return {}

    # Pre-allocate columnar arrays
    record_cols = get_record_columns()
    arrays: dict[str, np.ndarray[Any, Any]] = {
        col: np.empty(n, dtype=object) for col in record_cols
    }

    for i, vuln in enumerate(vulnerabilities):
        cve_data = vuln.get("cve", {})

        # Identity
        arrays["id"][i] = str(uuid4())
        arrays["cve_id"][i] = cve_data.get("id")

        # Description (extract English)
        description = None
        for desc in cve_data.get("descriptions", []):
            if desc.get("lang") == "en":
                description = desc.get("value")
                break
        arrays["description"][i] = description

        # NVD Temporal
        arrays["nvd_published_at"][i] = parse_timestamp(cve_data.get("published"))
        arrays["nvd_modified_at"][i] = parse_timestamp(cve_data.get("lastModified"))
        arrays["nvd_status"][i] = extract_nvd_status(cve_data)

        # Complex fields: Parse to dataclasses -> to_dict() -> JSON

        # CVSS scores
        cvss_entries = parse_cvss(cve_data.get("metrics", {}))
        cvss_dicts = [entry.to_dict() for entry in cvss_entries]
        arrays["cvss_scores"][i] = json.dumps(cvss_dicts) if cvss_dicts else "[]"

        # CWEs
        cwe_entries = parse_cwes(cve_data.get("weaknesses", []))
        cwe_dicts = [entry.to_dict() for entry in cwe_entries]
        arrays["cwes"][i] = json.dumps(cwe_dicts) if cwe_dicts else "[]"

        # CPEs
        cpe_entries = parse_cpes(cve_data.get("configurations", []))
        cpe_dicts = [entry.to_dict() for entry in cpe_entries]
        arrays["cpes"][i] = json.dumps(cpe_dicts) if cpe_dicts else "[]"

        # References
        ref_entries = parse_references(cve_data.get("references", []))
        ref_dicts = [entry.to_dict() for entry in ref_entries]
        arrays["references"][i] = json.dumps(ref_dicts) if ref_dicts else "[]"

        # Empty fields (not in NVD)
        arrays["epss_scores"][i] = "[]"
        arrays["kev"][i] = "null"
        arrays["exploits"][i] = "[]"
        arrays["capecs"][i] = "[]"
        arrays["stride_threats"][i] = "[]"
        arrays["tags"][i] = "{}"
        arrays["metadata"][i] = "{}"

    return arrays
