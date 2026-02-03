"""Tests for transform utility functions."""

from __future__ import annotations

from vulnflow.dataset import VulnDataset
from vulnflow.utils.transforms import is_async_transform, partition_transforms


class SyncTransform:
    def transform(self, dataset: VulnDataset) -> None:
        pass


class AsyncTransform:
    async def transform(self, dataset: VulnDataset) -> None:
        pass


class NotATransform:
    pass


def test_is_async_transform_detects_async():
    """is_async_transform returns True for async transform method."""
    assert is_async_transform(AsyncTransform())


def test_is_async_transform_rejects_sync():
    """is_async_transform returns False for sync transform method."""
    assert not is_async_transform(SyncTransform())


def test_is_async_transform_rejects_no_method():
    """is_async_transform returns False for objects without transform."""
    assert not is_async_transform(NotATransform())


def test_partition_transforms_separates_correctly():
    """partition_transforms separates async from sync transforms."""
    transforms = [AsyncTransform(), SyncTransform(), AsyncTransform()]

    async_t, sync_t = partition_transforms(transforms)

    assert len(async_t) == 2
    assert len(sync_t) == 1
    assert all(isinstance(t, AsyncTransform) for t in async_t)
    assert all(isinstance(t, SyncTransform) for t in sync_t)
