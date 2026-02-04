"""CPEPrimaryTransform - Extract primary CPE from cpes array.

Layer: Transform/Flatten
Dependencies: transforms.flatten.base

Selects the first CPE entry from the cpes array.
Extracts 4 scalar columns from the selected entry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from vulnflow.registry import FieldDefinition
from vulnflow.transforms.flatten.base import BaseFlatten
from vulnflow.utils import json

if TYPE_CHECKING:
    from vulnflow.dataset import VulnDataset


class CPEPrimaryTransform(BaseFlatten):
    """Extract primary (first) CPE and its components.

    Output columns (4):
    - cpe_vendor, cpe_product, cpe_version, cpe_fix_version
    """

    name: str = "CPEPrimaryTransform"
    input_column: str = "cpes"
    depends_on: list[str] = []

    OUTPUT_SCHEMA: list[FieldDefinition] = [
        FieldDefinition("cpe_vendor", "str", "CPE vendor", "CPEPrimaryTransform"),
        FieldDefinition("cpe_product", "str", "CPE product", "CPEPrimaryTransform"),
        FieldDefinition("cpe_version", "str", "CPE version", "CPEPrimaryTransform"),
        FieldDefinition(
            "cpe_fix_version",
            "str",
            "CPE fix version (versionEndExcluding)",
            "CPEPrimaryTransform",
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

        # Set all columns
        for name, arr in out.items():
            dataset.set_column(name, arr)
