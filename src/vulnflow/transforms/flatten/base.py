"""BaseFlatten - Abstract base for flatten transforms.

Layer: Transform/Flatten
Dependencies: transforms.base

NOTE: Inherits from BaseComputed because flatten transforms are:
- Synchronous (not async I/O)
- Pure computation (selecting items from arrays)
- No external data fetching
- No confidence thresholds
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from vulnflow.transforms.base import BaseComputed

if TYPE_CHECKING:
    from vulnflow.dataset import VulnDataset


class BaseFlatten(BaseComputed):
    """Abstract base class for flatten transforms.

    Flatten transforms select the "best" item from a nested array
    and extract scalar columns from it.
    """

    # Subclass must define the input column containing the array
    input_column: str
    depends_on: list[str] = []  # Default: no dependencies (can override)

    def __init__(self) -> None:
        """Initialize flatten transform."""
        super().__init__()  # MUST call first per CLAUDE.md

    @abstractmethod
    def transform(self, dataset: VulnDataset) -> None:
        """Select best item from input_column array and add scalar columns."""
        ...
