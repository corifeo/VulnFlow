"""
Transform Base - Protocols for enrichers and transforms

Provides:
- Enricher: Protocol for async enrichment (I/O-bound, fetches external data)
- Transform: Protocol for sync transforms (CPU-bound, computes derived fields)

Layer: Transform
Dependencies: None (forward references)
Used by: enrichers/, computed/
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from vulnflow.dataset import VulnDataset


@runtime_checkable
class Enricher(Protocol):
    """Protocol for async enrichers that fetch external data."""

    name: str

    async def transform(self, dataset: VulnDataset) -> None:
        """Enrich dataset with external data (async, I/O-bound)."""
        ...


@runtime_checkable
class Transform(Protocol):
    """Protocol for sync transforms that compute derived fields."""

    name: str
    depends_on: list[str]

    def transform(self, dataset: VulnDataset) -> None:
        """Transform dataset by computing derived fields (sync, CPU-bound)."""
        ...


@runtime_checkable
class Inference(Protocol):
    """Protocol for inference transforms that fill gaps with heuristics.

    Inferences:
    - Only populate missing fields (never overwrite existing data)
    - Use confidence threshold to filter low-quality inferences
    - Are CPU-bound, sync operations (no I/O)
    - Run after Enrichers, before Computed transforms
    """

    name: str
    confidence_threshold: float  # 0.0-1.0, minimum confidence to populate
    depends_on: list[str]

    def transform(self, dataset: VulnDataset) -> None:
        """Infer missing fields where confidence >= threshold."""
        ...
