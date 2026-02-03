"""Transform utility functions.

Provides:
- is_async_transform: Check if object has async transform() method
- partition_transforms: Separate async from sync transforms

Layer: Utils
Dependencies: None
Used by: dataset, pipeline
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


def is_async_transform(obj: Any) -> bool:
    """Check if object has async transform() method.

    Args:
        obj: Object to check

    Returns:
        True if has async transform(), False otherwise
    """
    return hasattr(obj, "transform") and inspect.iscoroutinefunction(obj.transform)


def partition_transforms(
    transforms: list[Any],
) -> tuple[list[Any], list[Any]]:
    """Partition transforms into async and sync groups.

    Separates in a single pass to avoid redundant checks.

    Args:
        transforms: List of transform objects

    Returns:
        Tuple of (async_transforms, sync_transforms)
    """
    async_transforms: list[Any] = []
    sync_transforms: list[Any] = []

    for t in transforms:
        if is_async_transform(t):
            async_transforms.append(t)
        else:
            sync_transforms.append(t)

    return async_transforms, sync_transforms
