"""Transform utility functions.

Provides:
- is_async_transform: Check if object has async transform() method
- partition_transforms: Separate async from sync transforms

Layer: Utility
Dependencies: None
Used by: dataset, pipeline

Usage Notes:
    These utilities enable mixed async/sync transform execution:
    - Async transforms (BaseEnricher) run concurrently with asyncio.gather()
    - Sync transforms (BaseComputed, BaseInference) run sequentially
    - Partitioning happens once per pipeline execution
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    pass

T = TypeVar("T")


def is_async_transform(obj: Any) -> bool:
    """Check if object has async transform() method.

    Used to distinguish BaseEnricher (async I/O-bound) from BaseComputed/
    BaseInference (sync CPU-bound) transforms.

    Args:
        obj: Object to check (typically a transform instance)

    Returns:
        True if obj.transform is an async coroutine function, False otherwise

    Examples:
        >>> class AsyncTransform:
        ...     async def transform(self, data):
        ...         return data
        >>> class SyncTransform:
        ...     def transform(self, data):
        ...         return data
        >>> is_async_transform(AsyncTransform())
        True
        >>> is_async_transform(SyncTransform())
        False
    """
    return hasattr(obj, "transform") and inspect.iscoroutinefunction(obj.transform)


def partition_transforms(
    transforms: list[T],
) -> tuple[list[T], list[T]]:
    """Partition transforms into async and sync groups.

    Separates in a single pass to avoid redundant checks. Order is preserved
    within each group.

    Args:
        transforms: List of transform objects (BaseEnricher, BaseComputed, BaseInference)

    Returns:
        Tuple of (async_transforms, sync_transforms) maintaining original order

    Examples:
        >>> async_t = AsyncTransform()
        >>> sync_t1 = SyncTransform()
        >>> sync_t2 = SyncTransform()
        >>> async_list, sync_list = partition_transforms([async_t, sync_t1, sync_t2])
        >>> len(async_list), len(sync_list)
        (1, 2)
    """
    async_transforms: list[T] = []
    sync_transforms: list[T] = []

    for t in transforms:
        if is_async_transform(t):
            async_transforms.append(t)
        else:
            sync_transforms.append(t)

    return async_transforms, sync_transforms
