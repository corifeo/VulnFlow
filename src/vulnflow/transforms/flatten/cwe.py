"""CWEPrimaryTransform - Extract primary CWE from cwes array.

Layer: Transform/Flatten
Dependencies: transforms.flatten.base

Selects the first CWE entry from the cwes array.
Extracts 2 scalar columns from the selected entry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from vulnflow.registry import FieldDefinition
from vulnflow.transforms.flatten.base import BaseFlatten
from vulnflow.utils import json

if TYPE_CHECKING:
    from vulnflow.dataset import VulnDataset


class CWEPrimaryTransform(BaseFlatten):
    """Extract primary (first) CWE and its components.

    Output columns (2):
    - cwe_id, cwe_name
    """

    name: str = "CWEPrimaryTransform"
    input_column: str = "cwes"
    depends_on: list[str] = []

    OUTPUT_SCHEMA: list[FieldDefinition] = [
        FieldDefinition("cwe_id", "str", "CWE identifier", "CWEPrimaryTransform"),
        FieldDefinition("cwe_name", "str", "CWE name", "CWEPrimaryTransform"),
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
            if not raw or raw == "[]" or raw == "null":
                continue
            cwes = json.loads(str(raw))
            if not cwes:
                continue

            # Selection: first entry
            selected = cwes[0]

            # Extract scalar fields
            out["cwe_id"][i] = selected.get("id")
            out["cwe_name"][i] = selected.get("name")

        # Set all columns
        for name, arr in out.items():
            dataset.set_column(name, arr)
