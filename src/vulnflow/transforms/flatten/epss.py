"""EPSSFlattenTransform - Select most recent EPSS score.

Layer: Transform/Flatten
Dependencies: transforms.flatten.base

Selects the EPSS entry with the latest date from the epss_scores array.
Extracts 4 scalar columns from the selected entry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from vulnflow.registry import FieldDefinition
from vulnflow.transforms.flatten.base import BaseFlatten
from vulnflow.utils import json

if TYPE_CHECKING:
    from vulnflow.dataset import VulnDataset


class EPSSFlattenTransform(BaseFlatten):
    """Select most recent EPSS score and extract components.

    Output columns (4):
    - epss_probability, epss_percentile, epss_date, epss_model
    """

    name: str = "EPSSFlattenTransform"
    input_column: str = "epss_scores"
    depends_on: list[str] = []

    OUTPUT_SCHEMA: list[FieldDefinition] = [
        FieldDefinition(
            "epss_probability", "float", "EPSS probability", "EPSSFlattenTransform"
        ),
        FieldDefinition(
            "epss_percentile", "float", "EPSS percentile", "EPSSFlattenTransform"
        ),
        FieldDefinition("epss_date", "str", "EPSS score date", "EPSSFlattenTransform"),
        FieldDefinition(
            "epss_model", "int", "EPSS model version", "EPSSFlattenTransform"
        ),
    ]

    def transform(self, dataset: VulnDataset) -> None:
        """Select most recent EPSS entry and extract all fields."""
        epss_json = dataset.get_column(self.input_column)
        n = len(dataset)

        # Initialize output arrays
        out: dict[str, np.ndarray] = {
            f.name: np.full(n, None, dtype=object) for f in self.OUTPUT_SCHEMA
        }

        for i, raw in enumerate(epss_json):
            if not raw or raw == "[]" or raw == "null":
                continue
            scores = json.loads(str(raw))
            if not scores:
                continue

            # Selection: max(date)
            def get_date(s: dict[str, Any]) -> str:
                date_val = s.get("date", "")
                return str(date_val) if date_val else ""

            selected = max(scores, key=get_date)

            # Extract scalar fields
            out["epss_probability"][i] = selected.get("probability")
            out["epss_percentile"][i] = selected.get("percentile")
            out["epss_date"][i] = selected.get("date")
            out["epss_model"][i] = selected.get("model")

        # Set all columns
        for name, arr in out.items():
            dataset.set_column(name, arr)
