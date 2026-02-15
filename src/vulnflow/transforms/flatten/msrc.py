"""MSRCFlattenTransform - Extract MSRC fields from metadata JSON.

Layer: Transform/Flatten
Dependencies: transforms.flatten.base

Extracts MSRC-specific fields from metadata.msrc to 5 scalar columns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from vulnflow.registry import FieldDefinition, register_transform
from vulnflow.transforms.flatten.base import BaseFlatten
from vulnflow.utils import json

if TYPE_CHECKING:
    from vulnflow.dataset import VulnDataset

# MSRC severity ordering (highest first)
_SEVERITY_ORDER: dict[str, int] = {
    "Critical": 4,
    "Important": 3,
    "Moderate": 2,
    "Low": 1,
}


@register_transform(default=False)
class MSRCFlattenTransform(BaseFlatten):
    """Extract MSRC metadata fields to scalars.

    Selection: metadata.msrc object

    Output columns (5):
    - msrc_exploited, msrc_publicly_disclosed, msrc_impact,
      msrc_max_severity, msrc_month_id
    """

    name: str = "MSRCFlattenTransform"
    input_column: str = "metadata"
    depends_on: list[str] = []

    OUTPUT_SCHEMA: list[FieldDefinition] = [
        FieldDefinition(
            "msrc_exploited",
            "bool",
            "Actively exploited per MSRC",
            "MSRCFlattenTransform",
        ),
        FieldDefinition(
            "msrc_publicly_disclosed",
            "bool",
            "Publicly disclosed per MSRC",
            "MSRCFlattenTransform",
        ),
        FieldDefinition(
            "msrc_impact",
            "str",
            "Impact type (RCE, EoP, etc.)",
            "MSRCFlattenTransform",
        ),
        FieldDefinition(
            "msrc_max_severity",
            "str",
            "Highest severity across products",
            "MSRCFlattenTransform",
        ),
        FieldDefinition(
            "msrc_month_id",
            "str",
            "MSRC release month ID",
            "MSRCFlattenTransform",
        ),
    ]

    def __init__(self) -> None:
        """Initialize MSRC flatten transform."""
        super().__init__()  # MUST call first per CLAUDE.md

    def transform(self, dataset: VulnDataset) -> None:
        """Extract MSRC metadata fields to scalars."""
        meta_json = dataset.get_column(self.input_column)
        n = len(dataset)

        # Initialize output arrays
        out: dict[str, np.ndarray[Any, Any]] = {
            f.name: np.full(n, None, dtype=object) for f in self.OUTPUT_SCHEMA
        }

        for i, raw in enumerate(meta_json):
            if not raw or raw == "{}":
                out["msrc_exploited"][i] = False
                out["msrc_publicly_disclosed"][i] = False
                continue

            meta = json.loads(str(raw))
            msrc = meta.get("msrc")
            if not msrc:
                out["msrc_exploited"][i] = False
                out["msrc_publicly_disclosed"][i] = False
                continue

            out["msrc_exploited"][i] = msrc.get("exploited", False)
            out["msrc_publicly_disclosed"][i] = msrc.get("publicly_disclosed", False)
            out["msrc_impact"][i] = msrc.get("impact") or None
            out["msrc_month_id"][i] = msrc.get("month_id") or None

            # Compute max severity from severity_by_product_id
            sev_map: dict[str, str] = msrc.get("severity_by_product_id", {})
            if sev_map:
                max_sev = max(
                    sev_map.values(),
                    key=lambda s: _SEVERITY_ORDER.get(s, 0),
                )
                out["msrc_max_severity"][i] = max_sev

        # Insert columns right after metadata
        dataset.insert_columns_after(self.input_column, out)
