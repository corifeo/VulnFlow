"""
Schema - VulnRecord and nested dataclasses
Defines core data structures for vulnerability records

Layer: Core data schema
Dependencies: None
Used by: dataset, extractors, transformers, loaders

Schema Version: 1.0.0 (complete from Phase 1, no incremental evolution)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4


@dataclass
class CWEEntry:
    """CWE weakness classification."""

    id: str  # "CWE-502"
    name: str | None = None  # "Deserialization of Untrusted Data"
    category_id: str | None = None  # "CWE-913"
    category_name: str | None = None  # "Improper Control of Dynamically-Managed Code Resources"
    abstraction_level: str | None = None  # "Category"|"Class"|"Base"|"Variant"|"Compound"


@dataclass
class CAPECEntry:
    """CAPEC attack pattern classification."""

    id: str  # "CAPEC-86"
    name: str | None = None  # "XSS via HTTP Headers"
    abstraction: str | None = None  # "Meta"|"Standard"|"Detailed"
    status: str | None = None  # "Draft"|"Stable"|"Deprecated"
    likelihood_of_attack: str | None = None  # "High"|"Medium"|"Low"
    typical_severity: str | None = None  # "Very High"|"High"|"Medium"|"Low"|"Very Low"


@dataclass
class STRIDEThreat:
    """STRIDE threat classification with provenance."""

    category: str  # "spoofing"|"tampering"|"repudiation"|"info_disclosure"|"dos"|"elevation"
    confidence: float  # 0.0-1.0
    source: str  # "capec_consequences"|"cwe_mapping"|"cvss_impact"|"description"|"cvss_vector"
    capec_ids: list[str] = field(default_factory=list)


@dataclass
class CPEEntry:
    """CPE product identification."""

    vendor: str  # "apache"
    product: str  # "log4j"
    version: str | None = None  # "2.14.1"
    raw: str | None = None  # Full CPE 2.3 string
    fix_version: str | None = None  # From versionEndExcluding


@dataclass
class ReferenceEntry:
    """External reference with tags."""

    url: str
    source: str | None = None  # "nvd", "vendor", "third-party"
    tags: list[str] = field(default_factory=list)  # ["Patch", "Vendor Advisory"]


@dataclass
class CVSSEntry:
    """CVSS scoring (multi-source support)."""

    version: str  # "3.1", "4.0"
    vector_string: str
    base_score: float
    base_severity: str  # "CRITICAL", "HIGH", "MEDIUM", "LOW", "NONE"
    source: str  # "nvd", "redhat", "microsoft"
    is_authoritative: bool = False  # True if type="Primary" in NVD
    exploitability_score: float | None = None
    impact_score: float | None = None
    source_url: str | None = None
    published_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass
class EPSSEntry:
    """EPSS probability (time-series support)."""

    probability: float
    percentile: float
    date: datetime
    model: int = 3  # EPSS model version (1, 2, 3)


@dataclass
class KEVEntry:
    """CISA Known Exploited Vulnerabilities (complete from start)."""

    date_added: datetime
    due_date: datetime | None
    vendor: str
    product: str
    vulnerability_name: str
    required_action: str
    known_ransomware_use: bool = False  # Complete field from Phase 1


@dataclass
class ExploitEntry:
    """Verified exploit with confidence scoring."""

    source: str  # "exploitdb", "metasploit", "github", "nuclei"
    id: str
    url: str | None = None
    description: str | None = None
    published_at: datetime | None = None
    confidence: float = 0.5  # 0.0-1.0 quality score


# Main VulnRecord (complete schema from Phase 1)


@dataclass
class VulnRecord:
    """Vulnerability record with all enrichment data.

    Schema ordered by relevance:
    1. Identity/Description - What is it?
    2. Metadata - When did we see it?
    3. NVD Temporal - What NVD provides
    4. CVD Timeline - Populated by EventInference (Phase 11)
    5. Scoring - How severe/likely?
    6. Exploitation - Is it being exploited?
    7. Classification - What type, what products?
    8. References - Supporting data
    9. Extensible - Custom fields
    """

    # Identity
    id: UUID = field(default_factory=uuid4)
    cve_id: str | None = None

    # Description
    description: str | None = None

    # NVD Temporal (what NVD provides)
    nvd_published_at: datetime | None = None
    nvd_modified_at: datetime | None = None
    nvd_status: str | None = None  # analyzed, awaiting, modified, rejected, deferred

    # Note: CVD Timeline (v/f/d/p/x/a_timestamp) is NOT part of VulnRecord.
    # EventInference (Phase 11) adds these columns directly to VulnDataset.

    # Scoring (multi-source, time-series)
    cvss_scores: list[CVSSEntry] = field(default_factory=list)
    epss_scores: list[EPSSEntry] = field(default_factory=list)

    # Exploitation
    kev: KEVEntry | None = None
    exploits: list[ExploitEntry] = field(default_factory=list)

    # Classification
    cwes: list[CWEEntry] = field(default_factory=list)
    cpes: list[CPEEntry] = field(default_factory=list)

    # References
    references: list[ReferenceEntry] = field(default_factory=list)

    # Extensible
    tags: dict[str, list[str]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
