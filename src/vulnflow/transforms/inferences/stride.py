"""
STRIDE Inference - Threat classification using tiered confidence model

Provides:
- STRIDEInference: Infers STRIDE threat categories from CAPEC, CWE, CVSS, description

Layer: Transform/Inferences
Dependencies: CAPECEnricher, CWEFlattenTransform, CVSSFlattenTransform
Used by: dataset.transform(), pipeline
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from vulnflow.dataset import VulnDataset

from vulnflow.registry import FieldDefinition, register_transform
from vulnflow.schema import STRIDEThreat
from vulnflow.transforms.base import BaseInference
from vulnflow.utils import json

logger = logging.getLogger(__name__)

# --- STRIDE category constants ---

SPOOFING = "spoofing"
TAMPERING = "tampering"
REPUDIATION = "repudiation"
INFO_DISCLOSURE = "info_disclosure"
DOS = "dos"
ELEVATION = "elevation"

STRIDE_CATEGORIES: list[str] = [
    SPOOFING, TAMPERING, REPUDIATION, INFO_DISCLOSURE, DOS, ELEVATION,
]

# --- Tier 1: CAPEC consequence mappings ---

SCOPE_TO_STRIDE: dict[str, str] = {
    "Authentication": SPOOFING,
    "Confidentiality": INFO_DISCLOSURE,
    "Integrity": TAMPERING,
    "Availability": DOS,
    "Access Control": ELEVATION,
    "Authorization": ELEVATION,
    "Non-Repudiation": REPUDIATION,
    "Accountability": REPUDIATION,
}

IMPACT_TO_STRIDE: dict[str, str] = {
    "Bypass Protection Mechanism": SPOOFING,
    "Modify Data": TAMPERING,
    "Alter Execution Logic": TAMPERING,
    "Hide Activities": REPUDIATION,
    "Read Data": INFO_DISCLOSURE,
    "Resource Consumption": DOS,
    "Unreliable Execution": DOS,
    "Gain Privileges": ELEVATION,
    "Execute Unauthorized Commands": ELEVATION,
}

# --- Tier 2: CWE → STRIDE static mapping ---

CWE_TO_STRIDE: dict[str, list[str]] = {
    # Spoofing
    "CWE-287": [SPOOFING],
    "CWE-290": [SPOOFING],
    "CWE-294": [SPOOFING],
    "CWE-295": [SPOOFING],
    "CWE-306": [SPOOFING],
    "CWE-798": [SPOOFING],
    "CWE-384": [SPOOFING],
    "CWE-613": [SPOOFING],
    # Tampering
    "CWE-20": [TAMPERING],
    "CWE-74": [TAMPERING],
    "CWE-79": [TAMPERING, INFO_DISCLOSURE],
    "CWE-89": [TAMPERING],
    "CWE-94": [TAMPERING, ELEVATION],
    "CWE-95": [TAMPERING],
    "CWE-96": [TAMPERING],
    "CWE-97": [TAMPERING],
    "CWE-98": [TAMPERING, ELEVATION],
    "CWE-352": [TAMPERING],
    "CWE-434": [TAMPERING, ELEVATION],
    "CWE-502": [TAMPERING, ELEVATION],
    "CWE-611": [TAMPERING, INFO_DISCLOSURE],
    "CWE-918": [TAMPERING, INFO_DISCLOSURE],
    # Repudiation
    "CWE-223": [REPUDIATION],
    "CWE-778": [REPUDIATION],
    "CWE-779": [REPUDIATION],
    # Information Disclosure
    "CWE-200": [INFO_DISCLOSURE],
    "CWE-209": [INFO_DISCLOSURE],
    "CWE-319": [INFO_DISCLOSURE],
    "CWE-327": [INFO_DISCLOSURE],
    "CWE-522": [INFO_DISCLOSURE],
    "CWE-532": [INFO_DISCLOSURE, REPUDIATION],
    "CWE-538": [INFO_DISCLOSURE],
    "CWE-548": [INFO_DISCLOSURE],
    "CWE-601": [INFO_DISCLOSURE, SPOOFING],
    "CWE-311": [INFO_DISCLOSURE],
    "CWE-312": [INFO_DISCLOSURE],
    "CWE-326": [INFO_DISCLOSURE],
    # Denial of Service
    "CWE-400": [DOS],
    "CWE-770": [DOS],
    "CWE-674": [DOS],
    "CWE-835": [DOS],
    "CWE-404": [DOS],
    "CWE-772": [DOS],
    "CWE-409": [DOS],
    # Elevation of Privilege
    "CWE-250": [ELEVATION],
    "CWE-266": [ELEVATION],
    "CWE-269": [ELEVATION],
    "CWE-284": [ELEVATION],
    "CWE-639": [ELEVATION],
    "CWE-862": [ELEVATION],
    "CWE-863": [ELEVATION],
    "CWE-732": [ELEVATION],
    # Memory safety (multi-impact)
    "CWE-787": [TAMPERING, DOS, ELEVATION],
    "CWE-416": [TAMPERING, DOS, ELEVATION],
    "CWE-476": [DOS],
    "CWE-121": [TAMPERING, DOS, ELEVATION],
    "CWE-122": [TAMPERING, DOS, ELEVATION],
    "CWE-125": [INFO_DISCLOSURE, DOS],
    "CWE-190": [TAMPERING, DOS],
    "CWE-415": [TAMPERING, DOS, ELEVATION],
    "CWE-843": [TAMPERING, ELEVATION],
    "CWE-120": [TAMPERING, DOS, ELEVATION],
    "CWE-119": [TAMPERING, DOS, ELEVATION],
    "CWE-131": [TAMPERING, DOS],
}

# --- Tier 4: Description keyword patterns ---

DESCRIPTION_PATTERNS: dict[str, list[str]] = {
    SPOOFING: [
        r"spoof",
        r"impersonat",
        r"authenticat\w*\s+bypass",
        r"forge\w*\s+token",
        r"bypass\w*\s+authenticat",
    ],
    TAMPERING: [
        r"tamper",
        r"inject",
        r"unauthoriz\w*\s+modif",
        r"manipulat\w*\s+input",
    ],
    REPUDIATION: [
        r"log\w*\s+(?:missing|insufficient|omit)",
        r"audit\w*\s+(?:fail|bypass)",
    ],
    INFO_DISCLOSURE: [
        r"disclos",
        r"expos\w*\s+(?:sensitiv|inform|data)",
        r"(?:information|data)\s+leak",
        r"sensitiv\w+\s+inform",
    ],
    DOS: [
        r"denial.of.service",
        r"\bcrash\b",
        r"exhaust",
        r"uncontrolled\s+resource",
        r"\bdos\b",
    ],
    ELEVATION: [
        r"privilege\w*\s+escalat",
        r"elevat\w*\s+privilege",
        r"gain\w*\s+(?:unauthoriz\w*\s+)?access",
    ],
}

# Pre-compile patterns for performance
_COMPILED_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    category: [re.compile(p, re.IGNORECASE) for p in patterns]
    for category, patterns in DESCRIPTION_PATTERNS.items()
}


@register_transform(default=False)
class STRIDEInference(BaseInference):
    """Infer STRIDE threat categories using a 5-tier confidence model.

    Tiers (high→low confidence):
    1. CAPEC consequences (0.95) — MITRE-curated scope/impact data
    2. CWE mapping (0.85) — static CWE→STRIDE lookup
    3. CVSS impact (0.75) — C/I/A metrics
    4. Description keywords (0.60) — pattern matching
    5. CVSS vector heuristics (0.45) — weak structural signals
    """

    name: str = "STRIDEInference"
    depends_on: list[str] = [
        "CAPECEnricher",
        "CWEFlattenTransform",
        "CVSSFlattenTransform",
    ]

    TIER_THRESHOLDS: dict[int, float] = {
        1: 0.95,  # CAPEC consequences (authoritative)
        2: 0.85,  # CWE → STRIDE static mapping
        3: 0.75,  # CVSS impact metrics
        4: 0.60,  # Description keywords
        5: 0.45,  # CVSS vector heuristics
    }

    OUTPUT_SCHEMA: list[FieldDefinition] = [
        FieldDefinition(
            "stride_threats",
            "str",
            "STRIDE threat categories (JSON)",
            "STRIDEInference",
        ),
    ]

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        overwrite: bool = False,
        tiers: list[int] | None = None,
        cache_dir: Path | str = ".cache",
    ) -> None:
        """Initialize STRIDE inference.

        Args:
            confidence_threshold: Minimum confidence to apply (gates tiers).
            overwrite: If True, re-infer even if stride_threats already set.
            tiers: Explicit tier list to run (overrides threshold gating).
            cache_dir: Directory containing .cache/mitre/capec_database.json.
        """
        super().__init__(confidence_threshold, overwrite, tiers)
        self.cache_dir = Path(cache_dir)

    def _load_capec_db(self) -> dict[str, dict[str, Any]]:
        """Load CAPEC database from disk cache."""
        cache_file = self.cache_dir / "mitre" / "capec_database.json"
        if cache_file.exists():
            result: dict[str, dict[str, Any]] = json.loads(cache_file.read_text())
            return result
        logger.warning("CAPEC database not found at %s", cache_file)
        return {}

    def transform(self, dataset: VulnDataset) -> None:
        """Infer STRIDE threats for each record using tiered confidence."""
        n = len(dataset)
        capec_db = self._load_capec_db()

        # Load input columns (all optional — tiers degrade gracefully)
        capecs_col = dataset.get_column_or_none("capecs")
        cwe_id_col = dataset.get_column_or_none("cwe_id")
        cvss_c = dataset.get_column_or_none("cvss_confidentiality")
        cvss_i = dataset.get_column_or_none("cvss_integrity")
        cvss_a = dataset.get_column_or_none("cvss_availability")
        cvss_pr = dataset.get_column_or_none("cvss_privileges_required")
        cvss_scope = dataset.get_column_or_none("cvss_scope")
        descriptions = dataset.get_column_or_none("description")

        # Idempotency: load existing if not overwriting
        stride_col = np.full(n, None, dtype=object)
        if not self.overwrite:
            existing = dataset.get_column_or_none("stride_threats")
            if existing is not None:
                stride_col[:] = existing

        tier_stats: dict[int, int] = dict.fromkeys(range(1, 6), 0)

        for i in range(n):
            if not self.overwrite and stride_col[i] is not None:
                continue

            # Accumulate: {category: (confidence, source, capec_ids)}
            # Higher-tier results take precedence (first-write wins per category)
            assigned: dict[str, tuple[float, str, list[str]]] = {}

            # Tier 1: CAPEC consequences
            if self._should_run(1) and capecs_col is not None:
                capec_results = self._infer_from_capec_consequences(
                    capecs_col[i], capec_db
                )
                for cat, cids in capec_results.items():
                    if cat not in assigned:
                        assigned[cat] = (0.95, "capec_consequences", cids)
                        tier_stats[1] += 1

            # Tier 2: CWE → STRIDE
            if self._should_run(2) and cwe_id_col is not None:
                cwe_cats = self._infer_from_cwe(cwe_id_col[i])
                for cat in cwe_cats:
                    if cat not in assigned:
                        assigned[cat] = (0.85, "cwe_mapping", [])
                        tier_stats[2] += 1

            # Tier 3: CVSS impact
            if self._should_run(3):
                cvss_cats = self._infer_from_cvss_impact(
                    cvss_c[i] if cvss_c is not None else None,
                    cvss_i[i] if cvss_i is not None else None,
                    cvss_a[i] if cvss_a is not None else None,
                )
                for cat in cvss_cats:
                    if cat not in assigned:
                        assigned[cat] = (0.75, "cvss_impact", [])
                        tier_stats[3] += 1

            # Tier 4: Description keywords
            if self._should_run(4) and descriptions is not None:
                desc_cats = self._infer_from_description(descriptions[i])
                for cat in desc_cats:
                    if cat not in assigned:
                        assigned[cat] = (0.60, "description", [])
                        tier_stats[4] += 1

            # Tier 5: CVSS vector heuristics
            if self._should_run(5):
                vec_cats = self._infer_from_cvss_vector(
                    cvss_pr[i] if cvss_pr is not None else None,
                    cvss_scope[i] if cvss_scope is not None else None,
                    cvss_i[i] if cvss_i is not None else None,
                )
                for cat in vec_cats:
                    if cat not in assigned:
                        assigned[cat] = (0.45, "cvss_vector", [])
                        tier_stats[5] += 1

            # Serialize
            if assigned:
                entries = [
                    STRIDEThreat(cat, conf, src, cids).to_dict()
                    for cat, (conf, src, cids) in sorted(assigned.items())
                ]
                stride_col[i] = json.dumps(entries)
            else:
                stride_col[i] = "[]"

        dataset.set_column("stride_threats", stride_col)

        # Per-tier debug stats
        tier_names = {
            1: "capec_consequences",
            2: "cwe_mapping",
            3: "cvss_impact",
            4: "description",
            5: "cvss_vector",
        }
        for tier, count in sorted(tier_stats.items()):
            if count > 0:
                logger.debug(
                    "Tier %d (%s): %d categories assigned",
                    tier,
                    tier_names.get(tier, str(tier)),
                    count,
                )

        logger.info(
            "STRIDEInference: %d records, tiers=%s, threshold=%.2f",
            n,
            self.tiers or "auto",
            self.confidence_threshold,
        )

    def _infer_from_capec_consequences(
        self,
        capecs_raw: object,
        capec_db: dict[str, dict[str, Any]],
    ) -> dict[str, list[str]]:
        """Tier 1: Map CAPEC consequences to STRIDE categories.

        Returns:
            Dict mapping STRIDE category to list of contributing CAPEC IDs.
        """
        if not capecs_raw or capecs_raw == "[]" or capecs_raw == "null":
            return {}

        try:
            capec_entries: list[dict[str, Any]] = json.loads(str(capecs_raw))
        except Exception:
            return {}

        result: dict[str, list[str]] = {}
        for entry in capec_entries:
            capec_id = entry.get("id")
            if not capec_id or capec_id not in capec_db:
                continue

            consequences = capec_db[capec_id].get("consequences")
            if not isinstance(consequences, list):
                continue

            for cons in consequences:
                scopes = cons.get("scopes", [])
                impacts = cons.get("impacts", [])

                for scope in scopes:
                    stride_cat = SCOPE_TO_STRIDE.get(scope)
                    if stride_cat:
                        result.setdefault(stride_cat, [])
                        if capec_id not in result[stride_cat]:
                            result[stride_cat].append(capec_id)

                for impact in impacts:
                    stride_cat = IMPACT_TO_STRIDE.get(impact)
                    if stride_cat:
                        result.setdefault(stride_cat, [])
                        if capec_id not in result[stride_cat]:
                            result[stride_cat].append(capec_id)

        return result

    def _infer_from_cwe(self, cwe_id: object) -> list[str]:
        """Tier 2: Map CWE ID to STRIDE categories via static lookup."""
        if not cwe_id or not isinstance(cwe_id, str):
            return []
        return CWE_TO_STRIDE.get(cwe_id, [])

    def _infer_from_cvss_impact(
        self,
        cvss_c: object,
        cvss_i: object,
        cvss_a: object,
    ) -> list[str]:
        """Tier 3: Map CVSS C/I/A impact metrics to STRIDE."""
        categories: list[str] = []
        if cvss_c == "H":
            categories.append(INFO_DISCLOSURE)
        if cvss_i == "H":
            categories.append(TAMPERING)
        if cvss_a == "H":
            categories.append(DOS)
        return categories

    def _infer_from_description(self, description: object) -> list[str]:
        """Tier 4: Match description keywords to STRIDE categories."""
        if not description or not isinstance(description, str):
            return []

        categories: list[str] = []
        for category, patterns in _COMPILED_PATTERNS.items():
            for pattern in patterns:
                if pattern.search(description):
                    categories.append(category)
                    break  # One match per category is sufficient
        return categories

    def _infer_from_cvss_vector(
        self,
        cvss_pr: object,
        cvss_scope: object,
        cvss_i: object,
    ) -> list[str]:
        """Tier 5: Weak heuristics from CVSS vector components."""
        categories: list[str] = []
        if cvss_pr == "N":
            categories.append(SPOOFING)
        if cvss_scope == "C" and cvss_i == "H":
            categories.append(ELEVATION)
        return categories
