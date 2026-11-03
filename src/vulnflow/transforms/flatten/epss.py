"""EPSSFlattenTransform - Select most recent EPSS score.

Layer: Transform/Flatten
Dependencies: transforms.flatten.base

Selects the EPSS entry with the latest date from the epss_scores array.
Extracts 5 scalar columns from the selected entry (including epss_bucket).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from vulnflow.registry import FieldDefinition, register_transform
from vulnflow.transforms.flatten.base import BaseFlatten
from vulnflow.utils import json

if TYPE_CHECKING:
    from vulnflow.dataset import VulnDataset

# EPSS bucket thresholds (descending order, first match wins)
EPSS_BUCKET_THRESHOLDS: tuple[tuple[float, str], ...] = (
    (0.5, "high"),
    (0.1, "medium"),
    (0.01, "low"),
)
EPSS_BUCKET_DEFAULT = "very_low"


@register_transform(default=True)
class EPSSFlattenTransform(BaseFlatten):
    """Select most recent EPSS score and extract components.

    Output columns (5):
    - epss_probability, epss_percentile, epss_date, epss_model, epss_bucket
    """

    name: str = "EPSSFlattenTransform"
    input_column: str = "epss_scores"
    depends_on: list[str] = ["EPSSEnricher"]

    def __init__(self) -> None:
        """Initialize EPSS flatten transform."""
        super().__init__()  # must run first

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
        FieldDefinition(
            "epss_bucket", "str", "EPSS risk bucket", "EPSSFlattenTransform"
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
            # Check for empty/null JSON (stored as strings in columnar format)
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
            prob = selected.get("probability")
            out["epss_probability"][i] = prob
            out["epss_percentile"][i] = selected.get("percentile")
            out["epss_date"][i] = selected.get("date")
            out["epss_model"][i] = selected.get("model")

            # Compute bucket from probability
            if prob is not None:
                bucket = EPSS_BUCKET_DEFAULT
                for threshold, label in EPSS_BUCKET_THRESHOLDS:
                    if float(prob) >= threshold:
                        bucket = label
                        break
                out["epss_bucket"][i] = bucket

        # Insert columns right after input_column (epss_scores)
        dataset.insert_columns_after(self.input_column, out)
