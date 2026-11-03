"""KEVFlattenTransform - Extract KEV fields from kev JSON object.

Layer: Transform/Flatten
Dependencies: transforms.flatten.base

Extracts KEV object to 8 scalar columns including is_kev boolean flag.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from vulnflow.registry import FieldDefinition, register_transform
from vulnflow.transforms.flatten.base import BaseFlatten
from vulnflow.utils import json
from vulnflow.utils.nvd import parse_timestamp

if TYPE_CHECKING:
    from vulnflow.dataset import VulnDataset


@register_transform(default=True)
class KEVFlattenTransform(BaseFlatten):
    """Extract KEV object fields to scalars.

    Selection: Single object (not array)

    Output columns (8):
    - is_kev, kev_date_added, kev_due_date, kev_vendor, kev_product,
      kev_vulnerability_name, kev_required_action, kev_ransomware
    """

    name: str = "KEVFlattenTransform"
    input_column: str = "kev"
    depends_on: list[str] = ["KEVEnricher"]

    OUTPUT_SCHEMA: list[FieldDefinition] = [
        FieldDefinition(
            "is_kev", "bool", "Is in CISA KEV catalog", "KEVFlattenTransform"
        ),
        FieldDefinition(
            "kev_date_added", "datetime", "Date added to KEV", "KEVFlattenTransform"
        ),
        FieldDefinition(
            "kev_due_date", "datetime", "Remediation due date", "KEVFlattenTransform"
        ),
        FieldDefinition("kev_vendor", "str", "Vendor name", "KEVFlattenTransform"),
        FieldDefinition("kev_product", "str", "Product name", "KEVFlattenTransform"),
        FieldDefinition(
            "kev_vulnerability_name",
            "str",
            "Vulnerability name",
            "KEVFlattenTransform",
        ),
        FieldDefinition(
            "kev_required_action", "str", "Required action", "KEVFlattenTransform"
        ),
        FieldDefinition(
            "kev_ransomware", "bool", "Known ransomware use", "KEVFlattenTransform"
        ),
    ]

    def __init__(self) -> None:
        """Initialize KEV flatten transform."""
        super().__init__()  # MUST call first per CLAUDE.md

    def transform(self, dataset: VulnDataset) -> None:
        """Extract KEV object fields to scalars."""
        kev_json = dataset.get_column(self.input_column)
        n = len(dataset)

        # Initialize output arrays
        out: dict[str, np.ndarray] = {
            f.name: np.full(n, None, dtype=object) for f in self.OUTPUT_SCHEMA
        }

        for i, raw in enumerate(kev_json):
            if not raw or raw == "null":
                out["is_kev"][i] = False
                continue

            kev = json.loads(str(raw))
            out["is_kev"][i] = True
            out["kev_date_added"][i] = parse_timestamp(kev.get("date_added"))
            out["kev_due_date"][i] = parse_timestamp(kev.get("due_date"))
            out["kev_vendor"][i] = kev.get("vendor")
            out["kev_product"][i] = kev.get("product")
            out["kev_vulnerability_name"][i] = kev.get("vulnerability_name")
            out["kev_required_action"][i] = kev.get("required_action")
            out["kev_ransomware"][i] = kev.get("known_ransomware_use", False)

        # Insert columns right after input_column (kev)
        dataset.insert_columns_after(self.input_column, out)
