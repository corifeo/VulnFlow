"""
Schema - Dataclasses for vulnerability records

Provides:
- VulnRecord: Main vulnerability record dataclass
- CVSSEntry: CVSS score entry
- EPSSEntry: EPSS probability entry
- KEVEntry: CISA KEV catalog entry
- ExploitEntry: Exploit reference entry

Layer: Core
Dependencies: None
Used by: dataset, extractors, loaders
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID


@dataclass
class CVSSEntry:
    """CVSS score from a specific source and version."""

    version: str  # "2.0", "3.0", "3.1", "4.0"
    vector: str
    base_score: float
    source: str  # "nvd", "cna", etc.


@dataclass
class EPSSEntry:
    """EPSS probability at a point in time."""

    date: datetime
    probability: float
    percentile: float


@dataclass
class KEVEntry:
    """CISA Known Exploited Vulnerabilities catalog entry."""

    date_added: datetime
    due_date: datetime | None
    vendor: str
    product: str
    vulnerability_name: str
    required_action: str


@dataclass
class ExploitEntry:
    """Reference to a public exploit."""

    source: str  # "exploitdb", "metasploit", "github", "nuclei"
    id: str
    url: str | None
    description: str | None


@dataclass
class VulnRecord:
    """Core vulnerability record schema."""

    # Identity
    id: UUID
    cve_id: str | None = None
    aliases: list[str] = field(default_factory=list)

    # Pipeline metadata
    first_seen_at: datetime = field(default_factory=datetime.now)
    last_updated_at: datetime = field(default_factory=datetime.now)
    sources: list[str] = field(default_factory=list)

    # Description
    title: str | None = None
    description: str | None = None

    # CVD Timestamps (nullable)
    v_timestamp: datetime | None = None  # Vendor awareness
    f_timestamp: datetime | None = None  # Fix ready
    d_timestamp: datetime | None = None  # Deployed
    p_timestamp: datetime | None = None  # Public disclosure
    x_timestamp: datetime | None = None  # Exploit public
    a_timestamp: datetime | None = None  # Attacks observed

    # Affected products
    cpes: list[str] = field(default_factory=list)
    vendors: list[str] = field(default_factory=list)
    products: list[str] = field(default_factory=list)

    # Weaknesses
    cwes: list[str] = field(default_factory=list)

    # Scores (nested, multi-source)
    cvss: list[CVSSEntry] = field(default_factory=list)
    epss: list[EPSSEntry] = field(default_factory=list)

    # Exploitation
    kev: KEVEntry | None = None
    exploits: list[ExploitEntry] = field(default_factory=list)

    # Threat intel (extensible)
    tags: dict[str, list[str]] = field(default_factory=dict)

    # Raw metadata (escape hatch)
    metadata: dict[str, Any] = field(default_factory=dict)
