"""
CVD Desiderata Transform - Compute 12 desiderata boolean flags

Provides:
- CVDDesiderataTransform: Computes desiderata flags from CVD state/timestamps

Layer: Transform/Computed
Dependencies: EventInference, CVDStateTransform

Desiderata measure ideal CVD event orderings:
- D1-D4: Vendor-centric (V before others)
- D5-D8: Public-centric (P ordering)
- D9-D12: Attack-centric (A ordering)
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from vulnflow.dataset import VulnDataset

from vulnflow.registry import register_transform


@register_transform
class CVDDesiderataTransform:
    """Compute 12 desiderata flags from CVD timestamps.

    Requires EventInference and CVDStateTransform to run first.

    Desiderata (simplified):
    - d1: V before P (vendor knows before public)
    - d2: V before X (vendor knows before exploit)
    - d3: V before A (vendor knows before attacks)
    - d4: F before P (fix ready before public)
    - d5: F before X (fix ready before exploit)
    - d6: F before A (fix ready before attacks)
    - d7: D before P (deployed before public)
    - d8: D before X (deployed before exploit)
    - d9: D before A (deployed before attacks)
    - d10: P before X (public before exploit)
    - d11: P before A (public before attacks)
    - d12: X before A (exploit before attacks)
    """

    name: str = "CVDDesiderataTransform"
    depends_on: list[str] = ["EventInference", "CVDStateTransform"]

    SCHEMA_VERSION: str = "1.0.0"
    OUTPUT_SCHEMA: list[dict[str, str]] = [
        {"name": "desiderata_d1", "type": "bool", "description": "V before P"},
        {"name": "desiderata_d2", "type": "bool", "description": "V before X"},
        {"name": "desiderata_d3", "type": "bool", "description": "V before A"},
        {"name": "desiderata_d4", "type": "bool", "description": "F before P"},
        {"name": "desiderata_d5", "type": "bool", "description": "F before X"},
        {"name": "desiderata_d6", "type": "bool", "description": "F before A"},
        {"name": "desiderata_d7", "type": "bool", "description": "D before P"},
        {"name": "desiderata_d8", "type": "bool", "description": "D before X"},
        {"name": "desiderata_d9", "type": "bool", "description": "D before A"},
        {"name": "desiderata_d10", "type": "bool", "description": "P before X"},
        {"name": "desiderata_d11", "type": "bool", "description": "P before A"},
        {"name": "desiderata_d12", "type": "bool", "description": "X before A"},
        {"name": "desiderata_count", "type": "int", "description": "Count of satisfied desiderata"},
    ]

    def transform(self, dataset: VulnDataset) -> None:
        """Compute desiderata flags from CVD timestamps.

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

        # Initialize desiderata arrays
        d1: list[bool | None] = [None] * n
        d2: list[bool | None] = [None] * n
        d3: list[bool | None] = [None] * n
        d4: list[bool | None] = [None] * n
        d5: list[bool | None] = [None] * n
        d6: list[bool | None] = [None] * n
        d7: list[bool | None] = [None] * n
        d8: list[bool | None] = [None] * n
        d9: list[bool | None] = [None] * n
        d10: list[bool | None] = [None] * n
        d11: list[bool | None] = [None] * n
        d12: list[bool | None] = [None] * n

        for i in range(n):
            v = self._get_ts(v_ts, i)
            f = self._get_ts(f_ts, i)
            d = self._get_ts(d_ts, i)
            p = self._get_ts(p_ts, i)
            x = self._get_ts(x_ts, i)
            a = self._get_ts(a_ts, i)

            # Compute desiderata (True if first event is before second, None if either missing)
            d1[i] = self._before(v, p)
            d2[i] = self._before(v, x)
            d3[i] = self._before(v, a)
            d4[i] = self._before(f, p)
            d5[i] = self._before(f, x)
            d6[i] = self._before(f, a)
            d7[i] = self._before(d, p)
            d8[i] = self._before(d, x)
            d9[i] = self._before(d, a)
            d10[i] = self._before(p, x)
            d11[i] = self._before(p, a)
            d12[i] = self._before(x, a)

        # Add desiderata columns
        dataset.set_column("desiderata_d1", np.array(d1, dtype=object))
        dataset.set_column("desiderata_d2", np.array(d2, dtype=object))
        dataset.set_column("desiderata_d3", np.array(d3, dtype=object))
        dataset.set_column("desiderata_d4", np.array(d4, dtype=object))
        dataset.set_column("desiderata_d5", np.array(d5, dtype=object))
        dataset.set_column("desiderata_d6", np.array(d6, dtype=object))
        dataset.set_column("desiderata_d7", np.array(d7, dtype=object))
        dataset.set_column("desiderata_d8", np.array(d8, dtype=object))
        dataset.set_column("desiderata_d9", np.array(d9, dtype=object))
        dataset.set_column("desiderata_d10", np.array(d10, dtype=object))
        dataset.set_column("desiderata_d11", np.array(d11, dtype=object))
        dataset.set_column("desiderata_d12", np.array(d12, dtype=object))

        # Compute desiderata count (how many are True)
        counts: list[int] = []
        for i in range(n):
            count = sum(
                1
                for val in [
                    d1[i], d2[i], d3[i], d4[i], d5[i], d6[i],
                    d7[i], d8[i], d9[i], d10[i], d11[i], d12[i]
                ]
                if val is True
            )
            counts.append(count)
        dataset.set_column("desiderata_count", np.array(counts, dtype=np.int32))

    def _get_ts(
        self, ts_array: np.ndarray | None, i: int
    ) -> datetime | None:
        """Get timestamp from array safely."""
        if ts_array is None or i >= len(ts_array):
            return None
        val = ts_array[i]
        return val if val is not None else None

    def _before(
        self, first: datetime | None, second: datetime | None
    ) -> bool | None:
        """Check if first timestamp is before second.

        Returns:
            True if first < second, False if first >= second, None if either is missing
        """
        if first is None or second is None:
            return None
        return first < second
