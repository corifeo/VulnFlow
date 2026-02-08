"""
Broken Enricher/Transform - Test fixtures for error handling

Simulates failures for testing strict/lenient modes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vulnflow.dataset import VulnDataset

from vulnflow.exceptions import EnricherError, TransformError


class BrokenEnricher:
    """Enricher that always raises EnricherError."""

    name: str = "BrokenEnricher"
    depends_on: list[str] = []

    async def transform(self, dataset: VulnDataset) -> None:
        """Raise error immediately."""
        raise EnricherError("BrokenEnricher: simulated failure")


class BrokenTransform:
    """Transform that always raises TransformError."""

    name: str = "BrokenTransform"
    depends_on: list[str] = []

    def transform(self, dataset: VulnDataset) -> None:
        """Raise error immediately."""
        raise TransformError("BrokenTransform: simulated failure")
