"""CWEFlattenTransform - Extract primary CWE from cwes array.

Layer: Transform/Flatten
Dependencies: transforms.flatten.base

Selects the first CWE entry from the cwes array.
Extracts 5 scalar columns from the selected entry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from vulnflow.registry import FieldDefinition, register_transform
from vulnflow.transforms.flatten.base import BaseFlatten
from vulnflow.utils import json

if TYPE_CHECKING:
    from vulnflow.dataset import VulnDataset


@register_transform(default=True)
class CWEFlattenTransform(BaseFlatten):
    """Select primary CWE (first entry) and extract all fields.

    Selection: First CWE from cwes array

    Output columns (5):
    - cwe_id, cwe_name, cwe_category_id, cwe_category_name, cwe_abstraction_level
    """

    name: str = "CWEFlattenTransform"
    input_column: str = "cwes"
    depends_on: list[str] = ["CWEEnricher"]

    def __init__(self) -> None:
        """Initialize CWE flatten transform."""
        super().__init__()  # must run first

    OUTPUT_SCHEMA: list[FieldDefinition] = [
        FieldDefinition("cwe_id", "str", "CWE identifier", "CWEFlattenTransform"),
        FieldDefinition("cwe_name", "str", "CWE weakness name", "CWEFlattenTransform"),
        FieldDefinition(
            "cwe_category_id", "str", "Parent CWE category", "CWEFlattenTransform"
        ),
        FieldDefinition(
            "cwe_category_name",
            "str",
            "Parent category name",
            "CWEFlattenTransform",
        ),
        FieldDefinition(
            "cwe_abstraction_level",
            "str",
            "CWE abstraction level",
            "CWEFlattenTransform",
        ),
    ]

    def transform(self, dataset: VulnDataset) -> None:
        """Extract first CWE entry and its fields."""
        cwes_json = dataset.get_column(self.input_column)
        n = len(dataset)

        # Initialize output arrays
        out: dict[str, np.ndarray] = {
            f.name: np.full(n, None, dtype=object) for f in self.OUTPUT_SCHEMA
        }

        for i, raw in enumerate(cwes_json):
            # Check for empty/null JSON (stored as strings in columnar format)
            if not raw or raw == "[]" or raw == "null":
                continue
            cwes = json.loads(str(raw))
            if not cwes:
                continue

            # Selection: first entry
            selected = cwes[0]

            # Extract all scalar fields
            out["cwe_id"][i] = selected.get("id")
            out["cwe_name"][i] = selected.get("name")
            out["cwe_category_id"][i] = selected.get("category_id")
            out["cwe_category_name"][i] = selected.get("category_name")
            out["cwe_abstraction_level"][i] = selected.get("abstraction_level")

        # Insert columns right after input_column (cwes)
        dataset.insert_columns_after(self.input_column, out)
