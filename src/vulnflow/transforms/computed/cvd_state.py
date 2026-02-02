"""
CVD State Transform - Compute CVD state from timestamps

Provides:
- CVDStateTransform: Computes 6-character state string from CVD timestamps

Layer: Transform/Computed
Dependencies: EventInference (provides CVD timestamps)

State format: "VFdPxa" where uppercase = event occurred, lowercase = not occurred
- V: Vendor awareness
- F: Fix ready
- D: Deployed (fix deployed)
- P: Public disclosure
- X: Exploit public
- A: Attacks observed
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from vulnflow.dataset import VulnDataset

from vulnflow.registry import register_transform


@register_transform
class CVDStateTransform:
    """Compute CVD state string from timestamps.

    Requires EventInference to run first (provides CVD timestamp columns).
    """

    name: str = "CVDStateTransform"
    depends_on: list[str] = ["EventInference"]

    SCHEMA_VERSION: str = "1.0.0"
    OUTPUT_SCHEMA: list[dict[str, str]] = [
        {"name": "cvd_state", "type": "str", "description": "6-character CVD state (e.g., 'VFdPxa')"},
    ]

    def transform(self, dataset: VulnDataset) -> None:
        """Compute CVD state from timestamp columns.

        Args:
            dataset: VulnDataset with CVD timestamp columns
        """
        n = len(dataset)

        # Get CVD timestamp columns (use public API)
        v_ts = dataset.get_column_or_none("v_timestamp")
        f_ts = dataset.get_column_or_none("f_timestamp")
        d_ts = dataset.get_column_or_none("d_timestamp")
        p_ts = dataset.get_column_or_none("p_timestamp")
        x_ts = dataset.get_column_or_none("x_timestamp")
        a_ts = dataset.get_column_or_none("a_timestamp")

        # Compute state strings
        states: list[str] = []
        for i in range(n):
            v = "V" if v_ts is not None and v_ts[i] is not None else "v"
            f = "F" if f_ts is not None and f_ts[i] is not None else "f"
            d = "D" if d_ts is not None and d_ts[i] is not None else "d"
            p = "P" if p_ts is not None and p_ts[i] is not None else "p"
            x = "X" if x_ts is not None and x_ts[i] is not None else "x"
            a = "A" if a_ts is not None and a_ts[i] is not None else "a"
            states.append(f"{v}{f}{d}{p}{x}{a}")

        dataset.set_column("cvd_state", np.array(states, dtype=object))
