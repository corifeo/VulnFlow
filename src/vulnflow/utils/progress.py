"""
Progress bar utilities for Jupyter/terminal display.

Provides centralized tqdm wrapper with VulnFlow defaults.
Auto-detects Jupyter vs terminal environment.

Provides:
- progress_bar: Factory function with VulnFlow defaults

Layer: Utils
Dependencies: tqdm
Used by: dataset.transform(), extractors

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
from typing import Any

from tqdm.auto import tqdm as _tqdm


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
