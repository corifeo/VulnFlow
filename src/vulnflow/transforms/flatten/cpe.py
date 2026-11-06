"""CPEFlattenTransform - Extract primary CPE from cpes array.

Layer: Transform/Flatten
Dependencies: transforms.flatten.base

Selects the first CPE entry from the cpes array.
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


@register_transform(default=True)
class CPEFlattenTransform(BaseFlatten):
    """Select primary CPE (first entry) and extract components.

    Selection: First CPE from cpes array

    Output columns (4):
    - cpe_vendor, cpe_product, cpe_version, cpe_fix_version
    """

    name: str = "CPEFlattenTransform"
    input_column: str = "cpes"
    depends_on: list[str] = []

    def __init__(self) -> None:
        """Initialize CPE flatten transform."""
        super().__init__()  # must run first

    OUTPUT_SCHEMA: list[FieldDefinition] = [
        FieldDefinition("cpe_vendor", "str", "CPE vendor", "CPEFlattenTransform"),
        FieldDefinition("cpe_product", "str", "CPE product", "CPEFlattenTransform"),
        FieldDefinition("cpe_version", "str", "CPE version", "CPEFlattenTransform"),
        FieldDefinition(
            "cpe_fix_version",
            "str",
            "CPE fix version (versionEndExcluding)",
            "CPEFlattenTransform",
        ),
    ]

    def transform(self, dataset: VulnDataset) -> None:
        """Extract first CPE entry and its fields."""
        cpes_json = dataset.get_column(self.input_column)
        n = len(dataset)

        # Initialize output arrays
        out: dict[str, np.ndarray] = {
            f.name: np.full(n, None, dtype=object) for f in self.OUTPUT_SCHEMA
        }

        for i, raw in enumerate(cpes_json):
            # Check for empty/null JSON (stored as strings in columnar format)
            if not raw or raw == "[]" or raw == "null":
                continue
            cpes = json.loads(str(raw))
            if not cpes:
                continue

            # Selection: first entry
            selected = cpes[0]

            # Extract scalar fields
            out["cpe_vendor"][i] = selected.get("vendor")
            out["cpe_product"][i] = selected.get("product")
            out["cpe_version"][i] = selected.get("version")
            out["cpe_fix_version"][i] = selected.get("fix_version")

        # Insert columns right after input_column (cpes)
        dataset.insert_columns_after(self.input_column, out)
