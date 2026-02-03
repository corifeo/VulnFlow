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

from vulnflow.registry import FieldDefinition, register_transform
from vulnflow.transforms.base import BaseComputed


@register_transform(default=True)
class CVDStateTransform(BaseComputed):
    """Compute CVD state string from timestamps.

    Requires EventInference to run first (provides CVD timestamp columns).
    """

    name: str = "CVDStateTransform"
    depends_on: list[str] = ["EventInference"]

    SCHEMA_VERSION: str = "1.0.0"
    OUTPUT_SCHEMA: list[FieldDefinition] = [
        FieldDefinition(
            "cvd_state",
            "str",
            "6-character CVD state (e.g., 'VFdPxa')",
            "CVDStateTransform",
        ),
        FieldDefinition(
            "pair_mask",
            "int16",
            "15-bit mask encoding all pairwise event orderings",
            "CVDStateTransform",
        ),
        FieldDefinition(
            "fix_lag_days", "float64", "Days from V to F", "CVDStateTransform"
        ),
        FieldDefinition(
            "disclosure_lag_days", "float64", "Days from V to P", "CVDStateTransform"
        ),
        FieldDefinition(
            "deployment_lag_days", "float64", "Days from F to D", "CVDStateTransform"
        ),
        FieldDefinition(
            "exploit_lag_days", "float64", "Days from P to X", "CVDStateTransform"
        ),
        FieldDefinition(
            "attack_lag_days", "float64", "Days from X to A", "CVDStateTransform"
        ),
        FieldDefinition(
            "public_to_attack_days", "float64", "Days from P to A", "CVDStateTransform"
        ),
    ]

    def transform(self, dataset: VulnDataset) -> None:
        """Compute CVD state, pair mask, and time deltas.

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

        # Compute state strings and pair masks
        states: list[str] = []
        pair_masks = np.zeros(n, dtype=np.int16)

        for i in range(n):
            # Build state string
            v = "V" if v_ts is not None and v_ts[i] is not None else "v"
            f = "F" if f_ts is not None and f_ts[i] is not None else "f"
            d = "D" if d_ts is not None and d_ts[i] is not None else "d"
            p = "P" if p_ts is not None and p_ts[i] is not None else "p"
            x = "X" if x_ts is not None and x_ts[i] is not None else "x"
            a = "A" if a_ts is not None and a_ts[i] is not None else "a"
            states.append(f"{v}{f}{d}{p}{x}{a}")

            # Compute 15-bit pair mask
            mask = 0

            # Get timestamps for this record
            v_val = v_ts[i] if v_ts is not None else None
            f_val = f_ts[i] if f_ts is not None else None
            d_val = d_ts[i] if d_ts is not None else None
            p_val = p_ts[i] if p_ts is not None else None
            x_val = x_ts[i] if x_ts is not None else None
            a_val = a_ts[i] if a_ts is not None else None

            # Encode all 15 pairwise comparisons
            if v_val is not None and f_val is not None and v_val < f_val:
                mask |= 1 << 0  # V<F
            if v_val is not None and d_val is not None and v_val < d_val:
                mask |= 1 << 1  # V<D
            if v_val is not None and p_val is not None and v_val < p_val:
                mask |= 1 << 2  # V<P
            if v_val is not None and x_val is not None and v_val < x_val:
                mask |= 1 << 3  # V<X
            if v_val is not None and a_val is not None and v_val < a_val:
                mask |= 1 << 4  # V<A
            if f_val is not None and d_val is not None and f_val < d_val:
                mask |= 1 << 5  # F<D
            if f_val is not None and p_val is not None and f_val < p_val:
                mask |= 1 << 6  # F<P
            if f_val is not None and x_val is not None and f_val < x_val:
                mask |= 1 << 7  # F<X
            if f_val is not None and a_val is not None and f_val < a_val:
                mask |= 1 << 8  # F<A
            if d_val is not None and p_val is not None and d_val < p_val:
                mask |= 1 << 9  # D<P
            if d_val is not None and x_val is not None and d_val < x_val:
                mask |= 1 << 10  # D<X
            if d_val is not None and a_val is not None and d_val < a_val:
                mask |= 1 << 11  # D<A
            if p_val is not None and x_val is not None and p_val < x_val:
                mask |= 1 << 12  # P<X
            if p_val is not None and a_val is not None and p_val < a_val:
                mask |= 1 << 13  # P<A
            if x_val is not None and a_val is not None and x_val < a_val:
                mask |= 1 << 14  # X<A

            pair_masks[i] = mask

        # Compute time deltas (in days)
        def compute_delta_days(
            ts1: np.ndarray | None, ts2: np.ndarray | None
        ) -> np.ndarray:
            """Compute time delta in days between two timestamp arrays."""
            deltas = np.full(n, np.nan, dtype=np.float64)
            if ts1 is not None and ts2 is not None:
                for i in range(n):
                    if ts1[i] is not None and ts2[i] is not None:
                        deltas[i] = (ts2[i] - ts1[i]).total_seconds() / 86400.0
            return deltas

        fix_lag_days = compute_delta_days(v_ts, f_ts)
        disclosure_lag_days = compute_delta_days(v_ts, p_ts)
        deployment_lag_days = compute_delta_days(f_ts, d_ts)
        exploit_lag_days = compute_delta_days(p_ts, x_ts)
        attack_lag_days = compute_delta_days(x_ts, a_ts)
        public_to_attack_days = compute_delta_days(p_ts, a_ts)

        # Set all columns
        dataset.set_column("cvd_state", np.array(states, dtype=object))
        dataset.set_column("pair_mask", pair_masks)
        dataset.set_column("fix_lag_days", fix_lag_days)
        dataset.set_column("disclosure_lag_days", disclosure_lag_days)
        dataset.set_column("deployment_lag_days", deployment_lag_days)
        dataset.set_column("exploit_lag_days", exploit_lag_days)
        dataset.set_column("attack_lag_days", attack_lag_days)
        dataset.set_column("public_to_attack_days", public_to_attack_days)
