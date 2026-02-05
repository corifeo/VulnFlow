"""CWEFlattenTransform - Extract primary CWE from cwes array.

Layer: Transform/Flatten
Dependencies: transforms.flatten.base

Selects the first CWE entry from the cwes array.
Extracts 1 scalar column from the selected entry.
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
    """Select primary CWE (first entry) and extract identifier.

    Selection: First CWE from cwes array

    Output columns (1):
    - cwe_id
    """

    name: str = "CWEFlattenTransform"
    input_column: str = "cwes"
    depends_on: list[str] = []

    def __init__(self) -> None:
        """Initialize CWE flatten transform."""
        super().__init__()  # MUST call first per CLAUDE.md

    OUTPUT_SCHEMA: list[FieldDefinition] = [
        FieldDefinition("cwe_id", "str", "CWE identifier", "CWEFlattenTransform"),
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

            # Extract scalar field
            out["cwe_id"][i] = selected.get("id")

        # Set all columns
        for name, arr in out.items():
            dataset.set_column(name, arr)
