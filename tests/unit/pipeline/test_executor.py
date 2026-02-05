"""
Pipeline Executor - Tests for stage execution logic
Tests: Extractors, transforms (sync/async), loaders, error handling

Layer: Core ETL
Dependencies: dataset, extractors, transforms, loaders
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import pytest

from vulnflow.dataset import VulnDataset
from vulnflow.executor import PipelineExecutor, get_stage_name
from vulnflow.extractors.base import BaseExtractor
from vulnflow.loaders.base import BaseLoader

# =============================================================================
# get_stage_name Tests
# =============================================================================


def test_get_stage_name_prefers_name_attribute():
    """get_stage_name() prefers the name attribute."""

    class TestStage:
        name = "custom_name"

    obj = TestStage()
    assert get_stage_name(obj) == "custom_name"


def test_get_stage_name_fallback_to_class_name():
    """get_stage_name() falls back to class name."""

    class TestStage:
        pass

    obj = TestStage()
    assert get_stage_name(obj) == "TestStage"


# =============================================================================
# run_extractors Tests
# =============================================================================


def test_run_extractors_merges_datasets():
    """run_extractors() merges multiple extractor outputs."""

    class MockExtractor(BaseExtractor):
        format = "mock"
        name = "MockExtractor"

        def __init__(self, cve_id: str):
            super().__init__()
            self.cve_id = cve_id

        def extract(self) -> VulnDataset:
            ds = VulnDataset()
            ds._data = {"cve_id": np.array([self.cve_id])}
            return ds

    extractors = [MockExtractor("CVE-1"), MockExtractor("CVE-2")]
    dataset, successful, failed, errors = PipelineExecutor.run_extractors(extractors)

    assert len(dataset) == 2
    assert successful == ["MockExtractor", "MockExtractor"]
    assert failed == []
    assert errors == []


def test_run_extractors_raises_on_failure():
    """run_extractors() raises immediately on extractor failure."""

    class FailingExtractor(BaseExtractor):
        format = "fail"
        name = "FailingExtractor"

        def extract(self) -> VulnDataset:
            raise RuntimeError("Extractor failed")

    extractors = [FailingExtractor()]

    with pytest.raises(RuntimeError, match="Extractor failed"):
        PipelineExecutor.run_extractors(extractors)


def test_run_extractors_empty_list():
    """run_extractors() handles empty extractor list."""
    dataset, successful, failed, errors = PipelineExecutor.run_extractors([])

    assert len(dataset) == 0
    assert successful == []
    assert failed == []
    assert errors == []


# =============================================================================
# run_async_transforms Tests
# =============================================================================


@pytest.mark.asyncio
async def test_run_async_transforms_concurrent():
    """run_async_transforms() runs transforms concurrently."""
    import time

    class SlowTransform:
        name = "SlowTransform"
        depends_on: list[str] = []

        async def transform(self, dataset: VulnDataset) -> None:
            await asyncio.sleep(0.1)

    dataset = VulnDataset()
    transforms = [SlowTransform() for _ in range(3)]

    start = time.time()
    successful, failed, errors = await PipelineExecutor.run_async_transforms(
        dataset, transforms, strict=True
    )
    elapsed = time.time() - start

    # Should take ~0.1s (concurrent) not ~0.3s (sequential)
    assert elapsed < 0.2
    assert len(successful) == 3
    assert failed == []
    assert errors == []


@pytest.mark.asyncio
async def test_run_async_transforms_strict_raises():
    """run_async_transforms() in strict mode raises on error."""

    class FailingTransform:
        name = "FailingTransform"

        async def transform(self, dataset: VulnDataset) -> None:
            raise ValueError("Transform failed")

    dataset = VulnDataset()
    transforms = [FailingTransform()]

    with pytest.raises(ValueError, match="Transform failed"):
        await PipelineExecutor.run_async_transforms(dataset, transforms, strict=True)


@pytest.mark.asyncio
async def test_run_async_transforms_lenient_collects_errors():
    """run_async_transforms() in lenient mode collects errors."""

    class FailingTransform:
        name = "FailingTransform"

        async def transform(self, dataset: VulnDataset) -> None:
            raise ValueError("Transform failed")

    class GoodTransform:
        name = "GoodTransform"

        async def transform(self, dataset: VulnDataset) -> None:
            pass

    dataset = VulnDataset()
    transforms = [FailingTransform(), GoodTransform()]

    successful, failed, errors = await PipelineExecutor.run_async_transforms(
        dataset, transforms, strict=False
    )

    assert successful == ["GoodTransform"]
    assert failed == ["FailingTransform"]
    assert len(errors) == 1
    assert isinstance(errors[0], ValueError)


@pytest.mark.asyncio
async def test_run_async_transforms_empty_list():
    """run_async_transforms() handles empty transform list."""
    dataset = VulnDataset()
    successful, failed, errors = await PipelineExecutor.run_async_transforms(
        dataset, [], strict=True
    )

    assert successful == []
    assert failed == []
    assert errors == []


# =============================================================================
# run_sync_transforms Tests
# =============================================================================


def test_run_sync_transforms_sequential():
    """run_sync_transforms() runs transforms sequentially."""
    call_order = []

    class TrackingTransform:
        def __init__(self, name: str):
            self.name = name

        def transform(self, dataset: VulnDataset) -> None:
            call_order.append(self.name)

    dataset = VulnDataset()
    transforms = [TrackingTransform("T1"), TrackingTransform("T2")]

    successful, failed, errors = PipelineExecutor.run_sync_transforms(
        dataset, transforms, strict=True
    )

    assert call_order == ["T1", "T2"]
    assert successful == ["T1", "T2"]
    assert failed == []
    assert errors == []


def test_run_sync_transforms_strict_raises():
    """run_sync_transforms() in strict mode raises on error."""

    class FailingTransform:
        name = "FailingTransform"

        def transform(self, dataset: VulnDataset) -> None:
            raise ValueError("Transform failed")

    dataset = VulnDataset()
    transforms = [FailingTransform()]

    with pytest.raises(ValueError, match="Transform failed"):
        PipelineExecutor.run_sync_transforms(dataset, transforms, strict=True)


def test_run_sync_transforms_lenient_continues_after_error():
    """run_sync_transforms() in lenient mode continues after errors."""

    class FailingTransform:
        name = "FailingTransform"

        def transform(self, dataset: VulnDataset) -> None:
            raise ValueError("Transform failed")

    class GoodTransform:
        name = "GoodTransform"

        def transform(self, dataset: VulnDataset) -> None:
            pass

    dataset = VulnDataset()
    transforms = [FailingTransform(), GoodTransform()]

    successful, failed, errors = PipelineExecutor.run_sync_transforms(
        dataset, transforms, strict=False
    )

    assert successful == ["GoodTransform"]
    assert failed == ["FailingTransform"]
    assert len(errors) == 1
    assert isinstance(errors[0], ValueError)


# =============================================================================
# run_loaders Tests
# =============================================================================


def test_run_loaders_returns_outputs():
    """run_loaders() collects outputs from all loaders."""

    class MockLoader1(BaseLoader):
        format = "mock"
        name = "loader1"

        def load(self, dataset: VulnDataset) -> Any:
            return {"path": "file1.json"}

    class MockLoader2(BaseLoader):
        format = "mock"
        name = "loader2"

        def load(self, dataset: VulnDataset) -> Any:
            return {"path": "file2.json"}

    dataset = VulnDataset()
    loaders = [MockLoader1(), MockLoader2()]

    outputs, successful, failed, errors = PipelineExecutor.run_loaders(dataset, loaders)

    assert len(outputs) == 2
    assert "loader1" in outputs
    assert "loader2" in outputs
    assert outputs["loader1"] == {"path": "file1.json"}
    assert outputs["loader2"] == {"path": "file2.json"}
    assert successful == ["loader1", "loader2"]
    assert failed == []
    assert errors == []


def test_run_loaders_raises_on_failure():
    """run_loaders() raises immediately on loader failure."""

    class FailingLoader(BaseLoader):
        format = "fail"
        name = "FailingLoader"

        def load(self, dataset: VulnDataset) -> Any:
            raise OSError("Loader failed")

    dataset = VulnDataset()
    loaders = [FailingLoader()]

    with pytest.raises(IOError, match="Loader failed"):
        PipelineExecutor.run_loaders(dataset, loaders)


def test_run_loaders_empty_list():
    """run_loaders() handles empty loader list."""
    dataset = VulnDataset()
    outputs, successful, failed, errors = PipelineExecutor.run_loaders(dataset, [])

    assert outputs == {}
    assert successful == []
    assert failed == []
    assert errors == []


def test_run_loaders_uses_name_attribute():
    """run_loaders() uses loader.name for output keys."""

    class CustomNameLoader(BaseLoader):
        format = "custom"
        name = "custom_output_key"

        def load(self, dataset: VulnDataset) -> Any:
            return {"result": "success"}

    dataset = VulnDataset()
    loaders = [CustomNameLoader()]

    outputs, successful, failed, errors = PipelineExecutor.run_loaders(dataset, loaders)

    assert "custom_output_key" in outputs
    assert outputs["custom_output_key"] == {"result": "success"}
