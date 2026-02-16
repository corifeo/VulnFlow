"""Stage Observability Mixin - Logging, timing, and progress for all pipeline stages.

Provides:
- StageObservability: Mixin inherited by BaseExtractor, BaseTransform, BaseLoader

Layer: Utils
Dependencies: utils.progress
Used by: extractors.base, transforms.base, loaders.base
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any

from vulnflow.utils.progress import progress_bar

__all__ = ["StageObservability"]


class StageObservability:
    """Mixin providing logging, timing, and progress to all pipeline stages.

    Base classes (BaseExtractor, BaseTransform, BaseLoader) inherit this mixin
    and call _init_observability() as the first line of __init__().

    Every stage automatically gets:
    - self.logger: Standard Python logger (named after the stage's module)
    - self._timings: Dict recording sub-operation durations
    - self._timed(name): Context manager to time a block
    - self._progress(total, desc, unit): Factory for progress bars
    """

    logger: logging.Logger
    _timings: dict[str, float]

    def _init_observability(self) -> None:
        """Initialize logger and timings. Call as first line of __init__."""
        self.logger = logging.getLogger(type(self).__module__)
        self._timings = {}

    @contextmanager
    def _timed(self, name: str) -> Iterator[None]:
        """Record wall-clock duration of a block in self._timings[name]."""
        t0 = perf_counter()
        try:
            yield
        finally:
            self._timings[name] = perf_counter() - t0

    def _progress(
        self,
        total: int | None = None,
        desc: str | None = None,
        unit: str = "items",
        **kwargs: Any,
    ) -> Any:
        """Create a progress bar using the stage's label/name as default desc."""
        label: str = str(
            getattr(self, "label", None) or getattr(self, "name", type(self).__name__)
        )
        return progress_bar(total=total, desc=desc or label, unit=unit, **kwargs)
