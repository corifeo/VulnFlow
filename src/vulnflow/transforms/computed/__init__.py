"""
Computed Transforms - Sync computed field transforms

Provides:
- CVDStateTransform: Computes 6-char CVD state from timestamps
- CVDDesiderataTransform: Computes 12 boolean metrics

Layer: Transform/Computed
Dependencies: base, schema
Used by: dataset.transform()
"""

from __future__ import annotations

from vulnflow.transforms.computed.cvd_state import CVDStateTransform
from vulnflow.transforms.computed.desiderata import CVDDesiderataTransform

__all__ = ["CVDStateTransform", "CVDDesiderataTransform"]
