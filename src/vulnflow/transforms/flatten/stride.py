"""STRIDEFlattenTransform - Extract STRIDE threat booleans from stride_threats array.

Layer: Transform/Flatten
Dependencies: transforms.flatten.base

Reads stride_threats JSON array and produces 6 boolean columns,
one per STRIDE category. Default is False (not None).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from vulnflow.registry import FieldDefinition, register_transform
from vulnflow.transforms.flatten.base import BaseFlatten
from vulnflow.utils import json

if TYPE_CHECKING:
    from vulnflow.dataset import VulnDataset

CATEGORY_TO_COLUMN: dict[str, str] = {
    "spoofing": "stride_spoofing",
    "tampering": "stride_tampering",
    "repudiation": "stride_repudiation",
    "info_disclosure": "stride_info_disclosure",
    "dos": "stride_dos",
    "elevation": "stride_elevation",
}


@register_transform(default=False)
class STRIDEFlattenTransform(BaseFlatten):
    """Extract STRIDE threat booleans from stride_threats JSON array.

    Output columns (6 booleans):
    - stride_spoofing, stride_tampering, stride_repudiation,
      stride_info_disclosure, stride_dos, stride_elevation
    """

    name: str = "STRIDEFlattenTransform"
    input_column: str = "stride_threats"
    depends_on: list[str] = ["STRIDEInference"]

    OUTPUT_SCHEMA: list[FieldDefinition] = [
        FieldDefinition(
            "stride_spoofing", "bool", "Spoofing threat", "STRIDEFlattenTransform"
        ),
        FieldDefinition(
            "stride_tampering", "bool", "Tampering threat", "STRIDEFlattenTransform"
        ),
        FieldDefinition(
            "stride_repudiation",
            "bool",
            "Repudiation threat",
            "STRIDEFlattenTransform",
        ),
        FieldDefinition(
            "stride_info_disclosure",
            "bool",
            "Information Disclosure threat",
            "STRIDEFlattenTransform",
        ),
        FieldDefinition(
            "stride_dos", "bool", "Denial of Service threat", "STRIDEFlattenTransform"
        ),
        FieldDefinition(
            "stride_elevation",
            "bool",
            "Elevation of Privilege threat",
            "STRIDEFlattenTransform",
        ),
    ]

    def __init__(self) -> None:
        """Initialize STRIDE flatten transform."""
        super().__init__()

    def transform(self, dataset: VulnDataset) -> None:
        """Extract STRIDE booleans from stride_threats array."""
        stride_json = dataset.get_column(self.input_column)
        n = len(dataset)

        # Default to False (not None) — absence of threat is meaningful
        out: dict[str, np.ndarray] = {
            f.name: np.full(n, False, dtype=object) for f in self.OUTPUT_SCHEMA
        }

        for i, raw in enumerate(stride_json):
            if not raw or raw == "[]" or raw == "null":
                continue
            threats = json.loads(str(raw))
            for threat in threats:
                col = CATEGORY_TO_COLUMN.get(threat.get("category"))
                if col:
                    out[col][i] = True

        dataset.insert_columns_after(self.input_column, out)
