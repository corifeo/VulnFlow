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
    likelihood_of_exploit: str | None = None  # "High"|"Medium"|"Low"
    consequences: list[dict[str, list[str]]] | None = None  # [{"scopes": [...], "impacts": [...]}]
    applicable_platforms: list[str] | None = None  # ["Java", "C", "Linux"]
    modes_of_introduction: list[str] | None = None  # ["Design", "Implementation"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "category_id": self.category_id,
            "category_name": self.category_name,
            "abstraction_level": self.abstraction_level,
            "likelihood_of_exploit": self.likelihood_of_exploit,
            "consequences": self.consequences,
            "applicable_platforms": self.applicable_platforms,
            "modes_of_introduction": self.modes_of_introduction,
        }


@dataclass
class CAPECEntry:
    """CAPEC attack pattern classification."""

    id: str  # "CAPEC-86"
    name: str | None = None  # "XSS via HTTP Headers"
    abstraction: str | None = None  # "Meta"|"Standard"|"Detailed"
    status: str | None = None  # "Draft"|"Stable"|"Deprecated"
    likelihood_of_attack: str | None = None  # "High"|"Medium"|"Low"
    typical_severity: str | None = None  # "Very High"|"High"|"Medium"|"Low"|"Very Low"
    skills_required: list[str] | None = None  # ["Low", "Medium"]
    consequences: list[dict[str, list[str]]] | None = None  # [{"scopes": [...], "impacts": [...]}]
    attack_techniques: list[str] | None = None  # MITRE ATT&CK IDs ["1110", "1548.001"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "abstraction": self.abstraction,
            "status": self.status,
            "likelihood_of_attack": self.likelihood_of_attack,
            "typical_severity": self.typical_severity,
            "skills_required": self.skills_required,
            "consequences": self.consequences,
            "attack_techniques": self.attack_techniques,
        }


@dataclass
class STRIDEThreat:
    """STRIDE threat classification with provenance."""

    category: str  # "spoofing"|"tampering"|"repudiation"|"info_disclosure"|"dos"|"elevation"
    confidence: float  # 0.0–1.0 (tier-based: 0.95/0.85/0.75/0.60/0.45)
    source: str  # "capec_consequences"|"cwe_mapping"|"cvss_impact"|"description"|"cvss_vector"
    capec_ids: list[str] = field(default_factory=list)  # ["CAPEC-86", "CAPEC-242"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "confidence": self.confidence,
            "source": self.source,
            "capec_ids": self.capec_ids,
        }


@dataclass
class CPEEntry:
    """CPE product identification."""

    vendor: str  # "apache"
    product: str  # "log4j"
    version: str | None = None  # "2.14.1"
    raw: str | None = None  # Full CPE 2.3 string
    fix_version: str | None = None  # From versionEndExcluding

    def to_dict(self) -> dict[str, Any]:
        return {
            "vendor": self.vendor,
            "product": self.product,
            "version": self.version,
            "raw": self.raw,
            "fix_version": self.fix_version,
        }


@dataclass
class ReferenceEntry:
    """External reference with tags."""

    url: str
    source: str | None = None  # "nvd", "vendor", "third-party"
    tags: list[str] = field(default_factory=list)  # ["Patch", "Vendor Advisory"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "source": self.source,
            "tags": self.tags,
        }


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "vector_string": self.vector_string,
            "base_score": self.base_score,
            "base_severity": self.base_severity,
            "source": self.source,
            "is_authoritative": self.is_authoritative,
            "exploitability_score": self.exploitability_score,
            "impact_score": self.impact_score,
            "source_url": self.source_url,
            "published_at": self.published_at,
            "updated_at": self.updated_at,
        }


@dataclass
class EPSSEntry:
    """EPSS probability (time-series support)."""

    probability: float  # 0.0–1.0 (e.g. 0.03765)
    percentile: float  # 0.0–1.0 (e.g. 0.92)
    date: datetime  # Score date (e.g. 2024-03-15)
    model: int = 3  # EPSS model version (1, 2, 3)

    def to_dict(self) -> dict[str, Any]:
        return {
            "probability": self.probability,
            "percentile": self.percentile,
            "date": self.date,
            "model": self.model,
        }


@dataclass
class KEVEntry:
    """CISA Known Exploited Vulnerabilities (complete from start)."""

    date_added: datetime  # "2022-01-10"
    due_date: datetime | None  # "2022-01-24"
    vendor: str  # "Apache"
    product: str  # "Log4j"
    vulnerability_name: str  # "Apache Log4j Remote Code Execution"
    required_action: str  # "Apply updates per vendor instructions."
    known_ransomware_use: bool = False  # True if used in ransomware campaigns

    def to_dict(self) -> dict[str, Any]:
        return {
            "date_added": self.date_added,
            "due_date": self.due_date,
            "vendor": self.vendor,
            "product": self.product,
            "vulnerability_name": self.vulnerability_name,
            "required_action": self.required_action,
            "known_ransomware_use": self.known_ransomware_use,
        }


@dataclass
class ExploitEntry:
    """Verified exploit with confidence scoring."""

    source: str  # "exploitdb", "metasploit", "github", "nuclei"
    id: str
    url: str | None = None
    description: str | None = None
    published_at: datetime | None = None
    confidence: float = 0.5  # 0.0–1.0 quality score (1.0 = verified exploit)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "id": self.id,
            "url": self.url,
            "description": self.description,
            "published_at": self.published_at,
            "confidence": self.confidence,
        }


# Main VulnRecord (complete schema from Phase 1)


@dataclass
class VulnRecord:
    """Vulnerability record with all enrichment data.

    Schema ordered by relevance:
    1. Identity/Description - What is it?
    2. Metadata - When did we see it?
    3. NVD Temporal - What NVD provides
    4. Scoring - How severe/likely?
    5. Exploitation - Is it being exploited?
    6. Classification - What type, what products?
    7. References - Supporting data
    8. Extensible - Custom fields
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

    # Scoring (multi-source, time-series)
    cvss_scores: list[CVSSEntry] = field(default_factory=list)
    epss_scores: list[EPSSEntry] = field(default_factory=list)

    # Exploitation
    kev: KEVEntry | None = None
    exploits: list[ExploitEntry] = field(default_factory=list)

    # Classification
    cwes: list[CWEEntry] = field(default_factory=list)
    cpes: list[CPEEntry] = field(default_factory=list)
    capecs: list[CAPECEntry] = field(default_factory=list)
    stride_threats: list[STRIDEThreat] = field(default_factory=list)

    # References
    references: list[ReferenceEntry] = field(default_factory=list)

    # Extensible
    tags: dict[str, list[str]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
