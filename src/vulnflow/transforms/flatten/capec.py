"""CAPECFlattenTransform - Extract primary CAPEC from capecs array.

Layer: Transform/Flatten
Dependencies: transforms.flatten.base

Selects the first CAPEC entry from the capecs array.
Extracts 4 scalar columns from the selected entry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from vulnflow.registry import FieldDefinition, register_transform
from vulnflow.transforms.flatten.base import BaseFlatten
from vulnflow.utils import json

if TYPE_CHECKING:
    from vulnflow.dataset import VulnDataset


@register_transform(default=False)
class CAPECFlattenTransform(BaseFlatten):
    """Select primary CAPEC (first entry) and extract key fields.

    Selection: First CAPEC from capecs array

    Output columns (4):
    - capec_id, capec_name, capec_likelihood, capec_severity
    """

    name: str = "CAPECFlattenTransform"
    input_column: str = "capecs"
    depends_on: list[str] = ["CAPECEnricher"]

    def __init__(self) -> None:
        """Initialize CAPEC flatten transform."""
        super().__init__()  # MUST call first per CLAUDE.md

    OUTPUT_SCHEMA: list[FieldDefinition] = [
        FieldDefinition("capec_id", "str", "CAPEC identifier", "CAPECFlattenTransform"),
        FieldDefinition(
            "capec_name", "str", "CAPEC attack pattern name", "CAPECFlattenTransform"
        ),
        FieldDefinition(
            "capec_likelihood",
            "str",
            "Likelihood of attack",
            "CAPECFlattenTransform",
        ),
        FieldDefinition(
            "capec_severity",
            "str",
            "Typical severity",
            "CAPECFlattenTransform",
        ),
    ]

    def transform(self, dataset: VulnDataset) -> None:
        """Extract first CAPEC entry and its fields."""
        capecs_json = dataset.get_column(self.input_column)
        n = len(dataset)

        # Initialize output arrays
        out: dict[str, np.ndarray] = {
            f.name: np.full(n, None, dtype=object) for f in self.OUTPUT_SCHEMA
        }

        for i, raw in enumerate(capecs_json):
            # Check for empty/null JSON (stored as strings in columnar format)
            if not raw or raw == "[]" or raw == "null":
                continue
            capecs = json.loads(str(raw))
            if not capecs:
                continue

            # Selection: first entry
            selected = capecs[0]

            # Extract scalar fields
            out["capec_id"][i] = selected.get("id")
            out["capec_name"][i] = selected.get("name")
            out["capec_likelihood"][i] = selected.get("likelihood_of_attack")
            out["capec_severity"][i] = selected.get("typical_severity")

        # Insert columns right after input_column (capecs)
        dataset.insert_columns_after(self.input_column, out)
