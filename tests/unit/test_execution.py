"""
Execution Engine - Unified transform execution
Tests: execute_transforms(), async concurrency, error handling

Layer: Core
Dependencies: topology, transforms
"""

from __future__ import annotations

import pytest
from vulnflow.dataset import VulnDataset
from vulnflow.execution import execute_transforms


@pytest.mark.asyncio
async def test_execute_transforms_runs_async_concurrently():
    """execute_transforms runs async transforms concurrently."""
    import asyncio
    import time

    call_log: list[tuple[str, float]] = []

    class AsyncTransform1:
        name = "async1"
        depends_on: list[str] = []

        async def transform(self, dataset):
            call_log.append(("async1_start", time.time()))
            await asyncio.sleep(0.1)
            call_log.append(("async1_end", time.time()))

    class AsyncTransform2:
        name = "async2"
        depends_on: list[str] = []

        async def transform(self, dataset):
            call_log.append(("async2_start", time.time()))
            await asyncio.sleep(0.1)
            call_log.append(("async2_end", time.time()))

    dataset = VulnDataset()
    transforms = [AsyncTransform1(), AsyncTransform2()]

    start = time.time()
    await execute_transforms(dataset, transforms, strict=True)
    elapsed = time.time() - start

    # Should take ~0.1s (concurrent), not ~0.2s (sequential)
    assert elapsed < 0.15, f"Async transforms ran sequentially ({elapsed:.2f}s)"

    # Both should start before either ends
    assert call_log[0][0] in ["async1_start", "async2_start"]
    assert call_log[1][0] in ["async1_start", "async2_start"]


def test_execute_transforms_sorts_by_dependencies():
    """execute_transforms applies topology sort before execution."""

    execution_order: list[str] = []

    class Transform1:
        name = "t1"
        depends_on: list[str] = []

        def transform(self, dataset):
            execution_order.append("t1")

    class Transform2:
        name = "t2"
        depends_on = ["t1"]

        def transform(self, dataset):
            execution_order.append("t2")

    dataset = VulnDataset()
    transforms = [Transform2(), Transform1()]  # Wrong order

    # Sync wrapper
    import asyncio

    asyncio.run(execute_transforms(dataset, transforms, strict=True))

    # Should execute in dependency order
    assert execution_order == ["t1", "t2"]
