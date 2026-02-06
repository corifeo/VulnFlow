"""Progress bar utilities for Jupyter/terminal display.

Provides centralized tqdm wrapper with VulnFlow defaults.
Auto-detects Jupyter vs terminal environment.

Provides:
- progress_bar: Factory function with VulnFlow defaults
- ProgressReporter: Manages tqdm bar + log lines for a pipeline phase
- _SuppressedProgress: Silent proxy returned when progress is suppressed
- _suppress_progress: ContextVar controlling suppression

Layer: Utils
Dependencies: tqdm

Configuration via environment variable:
    Set VULNFLOW_PROGRESS_STYLE to control display detail:
    - 'detailed': Full stats (rate, ETA, percentage, bar, counter)
    - 'simple': Minimal display (counter only, no bar or rate)
    - 'default': Balanced version (current behavior)

Example (Jupyter notebook):
    >>> import os
    >>> os.environ['VULNFLOW_PROGRESS_STYLE'] = 'detailed'
    >>>
    >>> from vulnflow.utils.progress import progress_bar
    >>> with progress_bar(total=100, desc="Loading CVEs") as pbar:
    ...     for i in range(100):
    ...         # Process item
    ...         pbar.update(1)
    # Detailed: "Loading CVEs... 100%|██████████| 100/100 [00:05<00:00, 20.00 records/s]"
    # Simple:   "Loading CVEs: 100/100"
    # Default:  "Loading CVEs... ██████████ 100/100 [00:05<00:00, 20.00 records/s]"

Example (disabling in tests):
    >>> import os
    >>> os.environ["TQDM_DISABLE"] = "1"
    >>> # Progress bars now silent
"""

from __future__ import annotations

import os
import time as _time
from contextvars import ContextVar, Token
from typing import Any

from tqdm.auto import tqdm as _tqdm

_suppress_progress: ContextVar[bool] = ContextVar(
    "_suppress_progress", default=False
)


class _SuppressedProgress:
    """Silent proxy returned when progress is suppressed."""

    def update(self, n: int = 1) -> None:  # noqa: ARG002
        pass

    def set_postfix_str(self, s: str, refresh: bool = True) -> None:  # noqa: ARG002
        pass

    def close(self) -> None:
        pass

    def __enter__(self) -> _SuppressedProgress:
        return self

    def __exit__(self, *args: Any) -> None:
        pass


class ProgressReporter:
    """Manages tqdm bar + log lines for a pipeline phase.

    Creates one tqdm bar for the phase. Prints log lines per stage.
    Suppresses inner progress_bar() calls via ContextVar.

    Usage:
        with ProgressReporter("Extracting data", total=4, unit="feeds") as rpt:
            rpt.stage_started("NVD-2023")
            ds = extractor.extract()
            rpt.stage_completed("NVD-2023", detail=f"{len(ds):,} CVEs")
    """

    def __init__(self, desc: str, total: int, unit: str) -> None:
        self._desc = desc
        self._total = total
        self._unit = unit
        self._bar: _tqdm[Any] | None = None
        self._token: Token[bool] | None = None
        self._start_times: dict[str, float] = {}

    def stage_started(self, name: str) -> None:
        """Record start time for this stage. No output."""
        self._start_times[name] = _time.perf_counter()

    def stage_completed(self, name: str, detail: str = "") -> None:
        """Print checkmark line with optional detail and duration, tick bar."""
        duration = self._elapsed(name)
        line = f"  \u2713 {name} ({self._format_inner(detail, duration)})"
        print(line)  # noqa: T201
        if self._bar is not None:
            self._bar.update(1)

    def stage_failed(self, name: str, error: str = "") -> None:
        """Print cross line with error and duration, tick bar."""
        duration = self._elapsed(name)
        line = f"  \u2717 {name}"
        if error:
            line += f" \u2014 {error}"
        line += f" ({duration:.1f}s)"
        print(line)  # noqa: T201
        if self._bar is not None:
            self._bar.update(1)

    def summary(self, text: str) -> None:
        """Print arrow summary line (no bar tick)."""
        print(f"  \u2192 {text}")  # noqa: T201

    def __enter__(self) -> ProgressReporter:
        self._bar = progress_bar(
            total=self._total,
            desc=self._desc,
            unit=self._unit,
        )
        self._token = _suppress_progress.set(True)
        return self

    def __exit__(self, *args: Any) -> None:
        if self._token is not None:
            _suppress_progress.reset(self._token)
            self._token = None
        if self._bar is not None:
            self._bar.close()
            self._bar = None

    def _elapsed(self, name: str) -> float:
        start = self._start_times.pop(name, None)
        if start is None:
            return 0.0
        return _time.perf_counter() - start

    def _format_inner(self, detail: str, duration: float) -> str:
        if detail:
            return f"{detail}, {duration:.1f}s"
        return f"{duration:.1f}s"


def progress_bar(
    iterable: Any = None,
    total: int | None = None,
    desc: str = "Processing",
    **kwargs: Any,
) -> _tqdm[Any]:
    """Create progress bar with VulnFlow defaults.

    Auto-detects Jupyter vs terminal using tqdm.auto.
    Style controlled by VULNFLOW_PROGRESS_STYLE environment variable.

    Args:
        iterable: Iterable to wrap (optional)
        total: Total iterations if not wrapping iterable
        desc: Description text
        **kwargs: Additional tqdm parameters (override style defaults)

    Returns:
        tqdm progress bar instance

    Environment Variables:
        VULNFLOW_PROGRESS_STYLE: 'detailed', 'simple', or 'default'
            - detailed: Full stats (rate, ETA, percentage, bar, counter)
            - simple: Minimal display (counter only, no bar or rate)
            - default: Balanced version (current behavior)

    Example:
        >>> import os
        >>> os.environ['VULNFLOW_PROGRESS_STYLE'] = 'detailed'
        >>> with progress_bar(total=100, desc="Loading") as pbar:
        ...     for i in range(100):
        ...         pbar.update(1)
    """
    if _suppress_progress.get(False):
        return _SuppressedProgress()  # type: ignore[return-value]

    style = os.getenv("VULNFLOW_PROGRESS_STYLE", "default").lower()

    # Style-specific defaults
    if style == "detailed":
        # Full stats: percentage, bar, counter, rate, ETA
        defaults: dict[str, Any] = {
            "unit": "records",
            "unit_scale": True,
            "leave": True,
            "bar_format": "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        }
    elif style == "simple":
        # Minimal: just counter, no bar or rate
        defaults = {
            "leave": True,
            "bar_format": "{desc}: {n_fmt}/{total_fmt}",
        }
    else:  # default
        # Balanced: bar, counter, rate (current behavior)
        defaults = {
            "unit": "records",
            "unit_scale": True,
            "leave": True,
        }

    # User kwargs override style defaults
    defaults.update(kwargs)

    return _tqdm(iterable=iterable, total=total, desc=desc, **defaults)
