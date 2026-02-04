"""CVSSFlattenTransform - Select authoritative CVSS score.

Layer: Transform/Flatten
Dependencies: transforms.flatten.base

Selects the best CVSS entry from the cvss_scores array based on:
1. is_authoritative=True (Primary in NVD)
2. First entry (fallback)

Extracts 15 scalar columns from the selected entry.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np

from vulnflow.registry import FieldDefinition
from vulnflow.transforms.flatten.base import BaseFlatten
from vulnflow.utils import json

if TYPE_CHECKING:
    from vulnflow.dataset import VulnDataset

# CVSS 3.x vector pattern
CVSS3_PATTERN = re.compile(
    r"CVSS:3\.[01]/AV:([NALP])/AC:([LH])/PR:([NLH])/UI:([NR])/S:([UC])/C:([NLH])/I:([NLH])/A:([NLH])"
)


class CVSSFlattenTransform(BaseFlatten):
    """Select authoritative CVSS score and extract components.

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
            if not raw or raw == "[]" or raw == "null":
                continue
            scores = json.loads(str(raw))
            if not scores:
                continue

            # Selection: is_authoritative=True > first
            selected = None
            for s in scores:
                if s.get("is_authoritative"):
                    selected = s
                    break
            if not selected:
                selected = scores[0]

            # Extract scalar fields
            out["cvss_version"][i] = selected.get("version")
            out["cvss_base_score"][i] = selected.get("base_score")
            out["cvss_base_severity"][i] = selected.get("base_severity")
            out["cvss_source"][i] = selected.get("source")
            out["cvss_vector_string"][i] = selected.get("vector_string")
            out["cvss_exploitability_score"][i] = selected.get("exploitability_score")
            out["cvss_impact_score"][i] = selected.get("impact_score")

            # Parse vector string
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

        # Set all columns
        for name, arr in out.items():
            dataset.set_column(name, arr)
