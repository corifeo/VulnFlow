"""Progress bar utilities for Jupyter/terminal display.

Provides centralized tqdm wrapper with VulnFlow defaults.
Auto-detects Jupyter vs terminal environment.

Provides:
- progress_bar: Factory function with VulnFlow defaults
- ProgressReporter: Manages tqdm bar + log lines for a pipeline phase
- ProgressMode: Enum controlling reporter output style (QUIET/BAR/LOG)
- _SuppressedProgress: Silent proxy returned when progress is suppressed
- _suppress_progress: ContextVar controlling suppression

Layer: Utils
Dependencies: tqdm

Configuration via environment variable:
    Set VULNFLOW_PROGRESS_STYLE to control display detail:
    - 'detailed': Full stats (rate, ETA, percentage, bar, counter)
    - 'simple': Minimal display (counter only, no bar or rate)
    - 'default': Balanced version (current behavior)

    Set VULNFLOW_PROGRESS_MODE to override mode detection:
    - 'quiet': dots only (./!)
    - 'bar': tqdm bars only, no text lines
    - 'log': text lines only, no tqdm bars

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
import sys
import time as _time
from contextvars import ContextVar, Token
from enum import Enum
from typing import Any

from tqdm.auto import tqdm as _tqdm

_suppress_progress: ContextVar[bool] = ContextVar("_suppress_progress", default=False)

# ASCII-safe glyphs (cp1252/Windows compatible)
_GLYPH_OK = "+"
_GLYPH_FAIL = "x"
_GLYPH_ARROW = ">"
_GLYPH_DASH = "--"


class ProgressMode(Enum):
    """Controls ProgressReporter output style.

    SILENT: no output at all -- for programmatic use (quiet=True)
    QUIET: dots only (./!) -- for minimal interactive feedback
    BAR: tqdm bars only, no text lines -- for notebooks/interactive
    LOG: text lines only (+/x/>), no tqdm bars -- for CI/piped output
    """

    SILENT = "silent"
    QUIET = "quiet"
    BAR = "bar"
    LOG = "log"


def _is_notebook() -> bool:
    """Detect if running inside a Jupyter/IPython notebook."""
    try:
        from IPython import get_ipython  # type: ignore[attr-defined]

        shell = get_ipython()  # type: ignore[no-untyped-call]
        if shell is None:
            return False
        return type(shell).__name__ == "ZMQInteractiveShell"
    except ImportError:
        return False


def _detect_mode() -> ProgressMode:
    """Resolve the progress mode using priority order.

    1. _suppress_progress ContextVar is True → QUIET
    2. VULNFLOW_PROGRESS_MODE env var → explicit override
    3. Auto-detect: notebook/tty → BAR, otherwise → LOG
    """
    if _suppress_progress.get(False):
        return ProgressMode.SILENT

    env_mode = os.getenv("VULNFLOW_PROGRESS_MODE", "").lower()
    if env_mode in ("silent", "quiet", "bar", "log"):
        return ProgressMode(env_mode)

    if _is_notebook() or (hasattr(sys.stdout, "isatty") and sys.stdout.isatty()):
        return ProgressMode.BAR

    return ProgressMode.LOG


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
    """Manages progress feedback for a pipeline phase.

    Adapts output based on ProgressMode:
    - QUIET: dots (./!) per stage, inner bars suppressed
    - BAR: outer tqdm bar + inner bars visible, no text lines
    - LOG: text lines (+/x/>) per stage, inner bars suppressed

    In LOG/QUIET mode, suppresses inner progress_bar() calls via ContextVar
    and TQDM_DISABLE env var (belt-and-suspenders for thread context
    propagation issues). In BAR mode, inner bars flow through naturally.

    Usage:
        with ProgressReporter("Processing feeds", total=4, unit="feeds") as rpt:
            rpt.stage_started("NVD-2023")
            ds = extractor.extract()
            rpt.stage_completed("NVD-2023", detail=f"{len(ds):,} CVEs")
    """

    def __init__(
        self,
        desc: str,
        total: int,
        unit: str,
        bar_format: str | None = None,
    ) -> None:
        self._desc = desc
        self._total = total
        self._unit = unit
        self._bar_format = bar_format
        self._bar: _tqdm[Any] | None = None
        self._token: Token[bool] | None = None
        self._start_times: dict[str, float] = {}
        self._mode: ProgressMode = ProgressMode.LOG
        self._dot_count: int = 0
        self.timings: dict[str, float] = {}
        self._old_tqdm_disable: str | None = None

    def stage_started(self, name: str) -> None:
        """Record start time. BAR mode: show active stage on bar."""
        self._start_times[name] = _time.perf_counter()
        if self._mode is ProgressMode.BAR and self._bar is not None:
            self._bar.set_postfix_str(name)

    def stage_completed(
        self,
        name: str,
        detail: str = "",
        duration: float | None = None,
        breakdown: str = "",
    ) -> None:
        """Report stage completion. Output depends on mode.

        QUIET: prints "." per stage.
        BAR: sets postfix on the bar (compact format), then updates counter.
        LOG: prints "+ name (detail -- breakdown, Xs)".

        Args:
            name: Stage name (must match a prior stage_started call).
            detail: Primary info for all modes (e.g., "42,362 CVEs").
            duration: Override duration in seconds. If None, uses internal timer.
            breakdown: Sub-timing parts, LOG mode only (e.g., "download 10.5s, populate 3.0s").
        """
        if duration is None:
            duration = self._elapsed(name)
        else:
            self._start_times.pop(name, None)

        self.timings[name] = duration

        if self._mode is ProgressMode.SILENT:
            pass  # No output at all
        elif self._mode is ProgressMode.QUIET:
            print(".", end="", flush=True)  # noqa: T201
            self._dot_count += 1
        elif self._mode is ProgressMode.BAR:
            if self._bar is not None:
                if detail:
                    self._bar.set_postfix_str(self._format_postfix(detail, duration))
                self._bar.update(1)
        else:  # LOG
            line = f"  {_GLYPH_OK} {name} ({self._format_log(detail, duration, breakdown)})"
            print(line)  # noqa: T201

    def stage_failed(
        self, name: str, error: str = "", duration: float | None = None
    ) -> None:
        """Report stage failure. Output depends on mode.

        QUIET: prints "!" per failed stage.
        BAR: writes failure via tqdm.write() and updates bar.
        LOG: prints "x name -- error (Xs)".

        Args:
            name: Stage name (must match a prior stage_started call).
            error: Optional error message.
            duration: Override duration in seconds. If None, uses internal timer.
        """
        if duration is None:
            duration = self._elapsed(name)
        else:
            self._start_times.pop(name, None)

        self.timings[name] = duration

        fail_line = f"  {_GLYPH_FAIL} {name}"
        if error:
            fail_line += f" {_GLYPH_DASH} {error}"
        fail_line += f" ({duration:.1f}s)"

        if self._mode is ProgressMode.SILENT:
            pass  # No output at all
        elif self._mode is ProgressMode.QUIET:
            print("!", end="", flush=True)  # noqa: T201
            self._dot_count += 1
        elif self._mode is ProgressMode.BAR:
            _tqdm.write(fail_line)
            if self._bar is not None:
                self._bar.update(1)
        else:  # LOG
            print(fail_line)  # noqa: T201

    def summary(self, text: str) -> None:
        """Show summary. BAR: sets postfix. LOG: prints arrow line.

        Suppressed in QUIET (dots suffice).
        """
        if self._mode is ProgressMode.BAR:
            if self._bar is not None:
                self._bar.set_postfix_str(text)
        elif self._mode is ProgressMode.LOG:
            print(f"  {_GLYPH_ARROW} {text}")  # noqa: T201

    def __enter__(self) -> ProgressReporter:
        self._mode = _detect_mode()

        if self._mode is ProgressMode.BAR:
            # BAR mode: outer bar + inner bars visible, no text.
            # Don't suppress inner progress_bar() calls — let them render.
            bar_kwargs: dict[str, Any] = {
                # Outer bars count discrete steps (feeds, transforms) —
                # unit_scale=False so tqdm shows "2/2" not "2.00/2.00".
                "unit_scale": False,
            }
            if self._bar_format is not None:
                bar_kwargs["bar_format"] = self._bar_format
            self._bar = progress_bar(
                total=self._total,
                desc=self._desc,
                unit=self._unit,
                **bar_kwargs,
            )
        else:
            # SILENT/QUIET/LOG: suppress inner bars via ContextVar + TQDM_DISABLE.
            # Belt-and-suspenders: ContextVar may not propagate to threads.
            self._old_tqdm_disable = os.environ.get("TQDM_DISABLE")
            os.environ["TQDM_DISABLE"] = "1"
            self._token = _suppress_progress.set(True)

        return self

    def __exit__(self, *args: Any) -> None:
        if self._mode not in (ProgressMode.BAR,):
            # Restore TQDM_DISABLE env var
            if self._old_tqdm_disable is None:
                os.environ.pop("TQDM_DISABLE", None)
            else:
                os.environ["TQDM_DISABLE"] = self._old_tqdm_disable

            if self._token is not None:
                _suppress_progress.reset(self._token)
                self._token = None

        if self._bar is not None:
            self._bar.set_postfix_str("")
            self._bar.close()
            self._bar = None

    def _elapsed(self, name: str) -> float:
        start = self._start_times.pop(name, None)
        if start is None:
            return 0.0
        return _time.perf_counter() - start

    def _format_log(self, detail: str, duration: float, breakdown: str = "") -> str:
        """Format for LOG mode: detail -- breakdown, Xs."""
        parts = detail
        if breakdown:
            parts = f"{parts} {_GLYPH_DASH} {breakdown}" if parts else breakdown
        if parts:
            return f"{parts}, {duration:.1f}s"
        return f"{duration:.1f}s"

    def _format_postfix(self, detail: str, duration: float) -> str:
        """Format for BAR mode postfix: detail in Xs."""
        if detail and duration > 0:
            return f"{detail} in {duration:.1f}s"
        return detail


def progress_bar(
    iterable: Any = None,
    total: int | None = None,
    desc: str = "Processing",
    **kwargs: Any,
) -> _tqdm[Any]:
    """Create progress bar with VulnFlow defaults.

    Auto-detects Jupyter vs terminal using tqdm.auto.
    Style controlled by VULNFLOW_PROGRESS_STYLE environment variable.

    Returns _SuppressedProgress (silent proxy) when suppressed via
    _suppress_progress ContextVar or TQDM_DISABLE env var.

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

    # Belt-and-suspenders: also check env var for threads that don't
    # inherit ContextVar. Prevents tqdm.notebook widget leaks.
    if os.environ.get("TQDM_DISABLE") == "1":
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
