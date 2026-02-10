"""CVSSFlattenTransform - Select authoritative CVSS score.

Layer: Transform/Flatten
Dependencies: transforms.flatten.base

Selects authoritative CVSS entries, falling back to best available Secondary:
- is_authoritative=True (Primary type in NVD data) preferred
- If no authoritative score exists, selects the highest-version Secondary score

This ensures cvss_source reflects the data source (nvd, redhat, etc.),
with CNA/ADP providers used as fallback when no Primary score exists.

Extracts 15 scalar columns from the selected entry.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import numpy as np

from vulnflow.registry import FieldDefinition, register_transform
from vulnflow.transforms.flatten.base import BaseFlatten
from vulnflow.utils import json

if TYPE_CHECKING:
    from vulnflow.dataset import VulnDataset

# CVSS 3.x vector pattern
CVSS3_PATTERN = re.compile(
    r"CVSS:3\.[01]/AV:([NALP])/AC:([LH])/PR:([NLH])/UI:([NR])/S:([UC])/C:([NLH])/I:([NLH])/A:([NLH])"
)

# CVSS 2.0 vector pattern
CVSS2_PATTERN = re.compile(
    r"AV:([NAL])/AC:([HML])/Au:([MSN])/C:([NPC])/I:([NPC])/A:([NPC])"
)

_VERSION_RANK: dict[str, int] = {"4.0": 3, "3.1": 2, "3.0": 1, "2.0": 0}


def _version_rank(entry: dict[str, Any]) -> int:
    """Rank CVSS entry by version for fallback selection."""
    return _VERSION_RANK.get(entry.get("version", ""), -1)


@register_transform(default=True)
class CVSSFlattenTransform(BaseFlatten):
    """Select authoritative CVSS score and extract components.

    Selects authoritative (Primary) CVSS scores, falling back to the
    highest-version Secondary score from CNA/ADP providers. If no
    authoritative score exists, selects the highest-version Secondary score.

    Output columns (15):
    - cvss_version, cvss_base_score, cvss_base_severity, cvss_source
    - cvss_vector_string, cvss_exploitability_score, cvss_impact_score
    - cvss_attack_vector, cvss_attack_complexity, cvss_privileges_required
    - cvss_user_interaction, cvss_scope
    - cvss_confidentiality, cvss_integrity, cvss_availability
    """

    name: str = "CVSSFlattenTransform"
    input_column: str = "cvss_scores"
    depends_on: list[str] = []

    def __init__(self) -> None:
        """Initialize CVSS flatten transform."""
        super().__init__()  # MUST call first per CLAUDE.md

    OUTPUT_SCHEMA: list[FieldDefinition] = [
        FieldDefinition("cvss_version", "str", "CVSS version", "CVSSFlattenTransform"),
        FieldDefinition(
            "cvss_base_score", "float", "CVSS base score", "CVSSFlattenTransform"
        ),
        FieldDefinition(
            "cvss_base_severity", "str", "CVSS severity", "CVSSFlattenTransform"
        ),
        FieldDefinition("cvss_source", "str", "CVSS source", "CVSSFlattenTransform"),
        FieldDefinition(
            "cvss_vector_string", "str", "CVSS vector string", "CVSSFlattenTransform"
        ),
        FieldDefinition(
            "cvss_exploitability_score",
            "float",
            "Exploitability subscore",
            "CVSSFlattenTransform",
        ),
        FieldDefinition(
            "cvss_impact_score", "float", "Impact subscore", "CVSSFlattenTransform"
        ),
        FieldDefinition(
            "cvss_attack_vector", "str", "Attack Vector (AV)", "CVSSFlattenTransform"
        ),
        FieldDefinition(
            "cvss_attack_complexity",
            "str",
            "Attack Complexity (AC)",
            "CVSSFlattenTransform",
        ),
        FieldDefinition(
            "cvss_privileges_required",
            "str",
            "Privileges Required (PR)",
            "CVSSFlattenTransform",
        ),
        FieldDefinition(
            "cvss_user_interaction",
            "str",
            "User Interaction (UI)",
            "CVSSFlattenTransform",
        ),
        FieldDefinition("cvss_scope", "str", "Scope (S)", "CVSSFlattenTransform"),
        FieldDefinition(
            "cvss_confidentiality",
            "str",
            "Confidentiality Impact (C)",
            "CVSSFlattenTransform",
        ),
        FieldDefinition(
            "cvss_integrity", "str", "Integrity Impact (I)", "CVSSFlattenTransform"
        ),
        FieldDefinition(
            "cvss_availability", "str", "Availability Impact (A)", "CVSSFlattenTransform"
        ),
    ]

    def transform(self, dataset: VulnDataset) -> None:
        """Select best CVSS entry and extract all fields."""
        cvss_json = dataset.get_column(self.input_column)
        n = len(dataset)

        # Initialize output arrays
        out: dict[str, np.ndarray] = {
            f.name: np.full(n, None, dtype=object) for f in self.OUTPUT_SCHEMA
        }

        for i, raw in enumerate(cvss_json):
            # Check for empty/null JSON (stored as strings in columnar format)
            if not raw or raw == "[]" or raw == "null":
                continue
            scores = json.loads(str(raw))
            if not scores:
                continue

            # Selection: prefer authoritative (Primary) score from NVD.
            # Fallback: highest-version Secondary score from CNA/ADP providers.
            selected = None
            fallback = None
            for s in scores:
                if s.get("is_authoritative"):
                    selected = s
                    break
                if fallback is None or _version_rank(s) > _version_rank(fallback):
                    fallback = s
            if selected is None:
                selected = fallback
            if not selected:
                continue

            # Extract scalar fields
            out["cvss_version"][i] = selected.get("version")
            out["cvss_base_score"][i] = selected.get("base_score")
            out["cvss_base_severity"][i] = selected.get("base_severity")
            out["cvss_source"][i] = selected.get("source")
            out["cvss_vector_string"][i] = selected.get("vector_string")
            out["cvss_exploitability_score"][i] = selected.get("exploitability_score")
            out["cvss_impact_score"][i] = selected.get("impact_score")

            # Read component fields directly from dict
            # (NVD provides them expanded in cvssData, parser stores them)
            version = selected.get("version", "")
            if version.startswith("4"):
                # v4.0: different field names, no scope
                out["cvss_attack_vector"][i] = selected.get("attack_vector")
                out["cvss_attack_complexity"][i] = selected.get("attack_complexity")
                out["cvss_privileges_required"][i] = selected.get("privileges_required")
                out["cvss_user_interaction"][i] = selected.get("user_interaction")
                # v4 has no scope — leave as None
                # v4 maps vuln_*_impact to generic columns
                out["cvss_confidentiality"][i] = selected.get("vuln_confidentiality_impact")
                out["cvss_integrity"][i] = selected.get("vuln_integrity_impact")
                out["cvss_availability"][i] = selected.get("vuln_availability_impact")
            elif selected.get("attack_vector") is not None:
                # v2/v3.x with component fields from dict
                out["cvss_attack_vector"][i] = selected.get("attack_vector")
                out["cvss_attack_complexity"][i] = selected.get("attack_complexity")
                out["cvss_privileges_required"][i] = selected.get("privileges_required")
                out["cvss_user_interaction"][i] = selected.get("user_interaction")
                out["cvss_scope"][i] = selected.get("scope")
                out["cvss_confidentiality"][i] = selected.get("confidentiality_impact")
                out["cvss_integrity"][i] = selected.get("integrity_impact")
                out["cvss_availability"][i] = selected.get("availability_impact")
            else:
                # Fallback: regex parse vector string (for old data without component fields)
                vector = selected.get("vector_string", "")
                if vector:
                    match = CVSS3_PATTERN.match(vector)
                    if match:
                        out["cvss_attack_vector"][i] = match.group(1)
                        out["cvss_attack_complexity"][i] = match.group(2)
                        out["cvss_privileges_required"][i] = match.group(3)
                        out["cvss_user_interaction"][i] = match.group(4)
                        out["cvss_scope"][i] = match.group(5)
                        out["cvss_confidentiality"][i] = match.group(6)
                        out["cvss_integrity"][i] = match.group(7)
                        out["cvss_availability"][i] = match.group(8)
                    else:
                        v2_match = CVSS2_PATTERN.match(vector)
                        if v2_match:
                            out["cvss_attack_vector"][i] = v2_match.group(1)
                            out["cvss_attack_complexity"][i] = v2_match.group(2)
                            # v2 Au != v3 PR; leave PR, UI, S as None
                            out["cvss_confidentiality"][i] = v2_match.group(4)
                            out["cvss_integrity"][i] = v2_match.group(5)
                            out["cvss_availability"][i] = v2_match.group(6)

        # Insert columns right after input_column (cvss_scores)
        dataset.insert_columns_after(self.input_column, out)
