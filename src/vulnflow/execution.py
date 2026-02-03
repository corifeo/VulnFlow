"""
Execution Engine - Unified transform execution

Provides:
- execute_transforms: Execute transforms with topology sort, async concurrency, error handling

Layer: Core
Dependencies: topology, transforms
Used by: dataset.transform(), pipeline.run()
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vulnflow.dataset import VulnDataset

from vulnflow.topology import topological_sort

logger = logging.getLogger(__name__)


async def execute_transforms(
    dataset: VulnDataset,
    transforms: list[Any],
    strict: bool = True,
) -> tuple[list[str], list[str], list[Exception]]:
    """Execute transforms with dependency resolution and async concurrency.

    Args:
        dataset: VulnDataset to transform
        transforms: List of transform instances
        strict: If True, raise on first error. If False, collect errors and continue.

    Returns:
        Tuple of (successful_names, failed_names, errors)
    """
    # Sort by dependencies
    sorted_transforms = topological_sort(transforms)

    # Separate async from sync
    async_transforms = [
        t
        for t in sorted_transforms
        if hasattr(t, "transform") and inspect.iscoroutinefunction(t.transform)
    ]
    sync_transforms = [
        t
        for t in sorted_transforms
        if hasattr(t, "transform") and not inspect.iscoroutinefunction(t.transform)
    ]

    successful: list[str] = []
    failed: list[str] = []
    errors: list[Exception] = []

    # Execute async transforms concurrently
    if async_transforms:
        if strict:
            # Strict mode: raise on first failure
            await asyncio.gather(*[t.transform(dataset) for t in async_transforms])
            for t in async_transforms:
                successful.append(t.name)
        else:
            # Lenient mode: collect exceptions
            results = await asyncio.gather(
                *[t.transform(dataset) for t in async_transforms],
                return_exceptions=True,
            )
            for t, result in zip(async_transforms, results, strict=True):
                if isinstance(result, Exception):
                    logger.warning("Transform %s failed: %s", t.name, result)
                    failed.append(t.name)
                    errors.append(result)
                else:
                    successful.append(t.name)

    # Execute sync transforms sequentially
    for t in sync_transforms:
        try:
            t.transform(dataset)
            successful.append(t.name)
        except Exception as e:
            logger.warning("Transform %s failed: %s", t.name, e)
            failed.append(t.name)
            errors.append(e)
            if strict:
                raise

    return successful, failed, errors


def execute_transforms_sync(
    dataset: VulnDataset,
    transforms: list[Any],
    strict: bool = True,
) -> tuple[list[str], list[str], list[Exception]]:
    """Synchronous wrapper for execute_transforms.

    Args:
        dataset: VulnDataset to transform
        transforms: List of transform instances
        strict: If True, raise on first error

    Returns:
        Tuple of (successful_names, failed_names, errors)
    """
    return asyncio.run(execute_transforms(dataset, transforms, strict))
