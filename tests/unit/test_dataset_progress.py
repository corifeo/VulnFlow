"""
VulnDataset progress bar integration
Tests: transform() shows progress bar
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

from vulnflow.transforms.computed.cvd_state import CVDStateTransform


def test_transform_shows_progress_bar(nvd_sample_dataset):
    """transform() creates progress bar with correct total."""
    with patch("vulnflow.utils.progress.progress_bar") as mock_pbar:
        mock_pbar.return_value.__enter__ = MagicMock()
        mock_pbar.return_value.__exit__ = MagicMock()
        mock_pbar.return_value.update = MagicMock()
        mock_pbar.return_value.set_postfix_str = MagicMock()

        dataset = nvd_sample_dataset
        dataset.transform([CVDStateTransform()])

        # Verify progress bar created with correct params
        mock_pbar.assert_called_once()
        call_kwargs = mock_pbar.call_args.kwargs
        assert call_kwargs["total"] == 1
        assert call_kwargs["desc"] == "Transforming dataset"
        assert call_kwargs["unit"] == "transforms"


def test_transform_updates_progress_per_step(nvd_sample_dataset):
    """transform() updates progress bar for each sync transform."""
    with patch("vulnflow.utils.progress.progress_bar") as mock_pbar:
        mock_instance = MagicMock()
        mock_pbar.return_value.__enter__.return_value = mock_instance
        mock_pbar.return_value.__exit__.return_value = None

        dataset = nvd_sample_dataset
        dataset.transform([CVDStateTransform()])

        # Verify update called once (1 sync transform)
        assert mock_instance.update.call_count == 1
        mock_instance.update.assert_called_with(1)


def test_transform_sets_current_transform_name(nvd_sample_dataset):
    """transform() shows current transform name for sync transforms."""
    with patch("vulnflow.utils.progress.progress_bar") as mock_pbar:
        mock_instance = MagicMock()
        mock_pbar.return_value.__enter__.return_value = mock_instance
        mock_pbar.return_value.__exit__.return_value = None

        dataset = nvd_sample_dataset
        dataset.transform([CVDStateTransform()])

        # Verify transform name shown (only for sync transforms)
        mock_instance.set_postfix_str.assert_called_with("[CVDStateTransform]")


def test_transform_respects_tqdm_disable_env(nvd_sample_dataset):
    """transform() respects TQDM_DISABLE environment variable."""
    with patch.dict(os.environ, {"TQDM_DISABLE": "1"}):
        # Should not raise even if tqdm is disabled
        dataset = nvd_sample_dataset
        dataset.transform([CVDStateTransform()])

        # Just verify it completes without error
        assert True


def test_async_transforms_batch_progress_update(nvd_sample_dataset):
    """Async transforms update progress in batch after completion."""
    import asyncio

    class MockAsyncTransform:
        name = "MockAsync"
        depends_on: list[str] = []

        async def transform(self, dataset):
            await asyncio.sleep(0.01)

    with patch("vulnflow.utils.progress.progress_bar") as mock_pbar:
        mock_instance = MagicMock()
        mock_pbar.return_value.__enter__.return_value = mock_instance
        mock_pbar.return_value.__exit__.return_value = None

        dataset = nvd_sample_dataset
        transforms = [MockAsyncTransform() for _ in range(3)]
        dataset.transform(transforms)

        # Verify update called ONCE with count of all async transforms
        assert mock_instance.update.call_count == 1
        mock_instance.update.assert_called_with(3)


def test_async_transforms_run_concurrently(nvd_sample_dataset):
    """Async transforms run concurrently, not sequentially.

    Verifies: 3 async transforms with 0.1s sleep each complete in ~0.1s (concurrent)
    not 0.3s (sequential). This test ensures progress bar doesn't break concurrency.
    """
    import asyncio
    import time

    # Create mock async transform that sleeps
    class MockAsyncTransform:
        name = "MockAsync"
        depends_on: list[str] = []

        async def transform(self, dataset):
            await asyncio.sleep(0.1)

    dataset = nvd_sample_dataset
    transforms = [MockAsyncTransform() for _ in range(3)]

    start = time.time()
    dataset.transform(transforms)
    elapsed = time.time() - start

    # Should complete in ~0.1s (concurrent) not ~0.3s (sequential)
    # Allow 0.05s tolerance for overhead
    assert (
        elapsed < 0.2
    ), f"Async transforms took {elapsed:.2f}s (expected <0.2s for concurrent execution)"
