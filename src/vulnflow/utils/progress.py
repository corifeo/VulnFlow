"""
Progress bar utilities for Jupyter/terminal display.

Provides centralized tqdm wrapper with VulnFlow defaults.
Auto-detects Jupyter vs terminal environment.

Provides:
- progress_bar: Factory function with VulnFlow defaults

Layer: Utils
Dependencies: tqdm
Used by: dataset.transform(), extractors

Example (Jupyter notebook):
    >>> from vulnflow.utils.progress import progress_bar
    >>>
    >>> with progress_bar(total=100, desc="Loading CVEs") as pbar:
    ...     for i in range(100):
    ...         # Process item
    ...         pbar.update(1)
    # Shows: "Loading CVEs... ██████████ 100/100 [00:05<00:00, 20.00 records/s]"

Example (disabling in tests):
    >>> import os
    >>> os.environ["TQDM_DISABLE"] = "1"
    >>> # Progress bars now silent
"""

from __future__ import annotations

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

    Args:
        iterable: Iterable to wrap (optional)
        total: Total iterations if not wrapping iterable
        desc: Description text
        **kwargs: Additional tqdm parameters (override defaults)

    Returns:
        tqdm progress bar instance

    Example:
        >>> with progress_bar(total=100, desc="Loading") as pbar:
        ...     for i in range(100):
        ...         pbar.update(1)
    """
    defaults: dict[str, Any] = {
        "unit": "records",
        "unit_scale": True,
        "leave": True,
    }
    defaults.update(kwargs)

    return _tqdm(iterable=iterable, total=total, desc=desc, **defaults)
