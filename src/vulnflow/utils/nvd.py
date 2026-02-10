"""
NVD Parsing Utilities - Parse NVD 2.0 JSON structures

Provides:
- parse_timestamp: ISO timestamp string to datetime
- extract_nvd_status: Normalize NVD vulnStatus field
- parse_cvss: Parse CVSS v2/v3.0/v3.1/v4.0 metrics
- parse_cwes: Parse CWE weaknesses
- parse_cpes: Parse CPE configurations
- parse_references: Parse reference entries

Layer: Utils
Dependencies: schema, utils.cvss, utils.cpe
Used by: NVDExtractor, future CVE.org extractor
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from vulnflow.schema import CPEEntry, CVSSEntry, CWEEntry, ReferenceEntry
from vulnflow.utils.cpe import parse_cpe23
from vulnflow.utils.cvss import extract_cvss_metadata, score_to_severity


def parse_timestamp(ts: str | None) -> datetime | None:
    """Parse ISO timestamp string to datetime."""
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


def parse_cvss(metrics: dict[str, list[dict[str, Any]]]) -> list[CVSSEntry]:
    """Parse CVSS metrics from NVD format.

    Handles version-specific field locations:
    - v4.0: baseSeverity NOT in cvssData, computed from score
    - v3.1/v3.0: baseSeverity in cvssData
    - v2.0: baseSeverity at metric level (not in cvssData)

    Args:
        metrics: NVD metrics dict (e.g., cve_data["metrics"])

    Returns:
        List of CVSSEntry dataclass instances
    """
    cvss_scores: list[CVSSEntry] = []

    # CVSS 4.0 - baseSeverity NOT in cvssData, compute from score
    for metric in metrics.get("cvssMetricV40", []):
        cvss_data = metric.get("cvssData", {})
        metadata = extract_cvss_metadata(metric)
        base_score = cvss_data.get("baseScore", 0.0)

        cvss_scores.append(
            CVSSEntry(
                version=cvss_data.get("version", "4.0"),
                vector_string=cvss_data.get("vectorString", ""),
                base_score=base_score,
                base_severity=score_to_severity(base_score),
                **metadata,
            )
        )

    # CVSS 3.1 - baseSeverity in cvssData
    for metric in metrics.get("cvssMetricV31", []):
        cvss_data = metric.get("cvssData", {})
        metadata = extract_cvss_metadata(metric)

        cvss_scores.append(
            CVSSEntry(
                version=cvss_data.get("version", "3.1"),
                vector_string=cvss_data.get("vectorString", ""),
                base_score=cvss_data.get("baseScore", 0.0),
                base_severity=cvss_data.get("baseSeverity", "NONE"),
                **metadata,
            )
        )

    # CVSS 3.0 - same structure as 3.1
    for metric in metrics.get("cvssMetricV30", []):
        cvss_data = metric.get("cvssData", {})
        metadata = extract_cvss_metadata(metric)

        cvss_scores.append(
            CVSSEntry(
                version=cvss_data.get("version", "3.0"),
                vector_string=cvss_data.get("vectorString", ""),
                base_score=cvss_data.get("baseScore", 0.0),
                base_severity=cvss_data.get("baseSeverity", "NONE"),
                **metadata,
            )
        )

    # CVSS 2.0 - baseSeverity at metric level (not in cvssData)
    for metric in metrics.get("cvssMetricV2", []):
        cvss_data = metric.get("cvssData", {})
        metadata = extract_cvss_metadata(metric)

        cvss_scores.append(
            CVSSEntry(
                version=cvss_data.get("version", "2.0"),
                vector_string=cvss_data.get("vectorString", ""),
                base_score=cvss_data.get("baseScore", 0.0),
                base_severity=metric.get(
                    "baseSeverity", "NONE"
                ),  # Different location!
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
