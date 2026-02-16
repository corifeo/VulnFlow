"""
Pipeline Executor - Tests for stage execution logic
Tests: Extractors (concurrent), transforms (phased), loaders, error handling

Layer: Core ETL
Dependencies: dataset, extractors, transforms, loaders
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vulnflow.dataset import VulnDataset
from vulnflow.executor import PipelineExecutor
from vulnflow.extractors.base import BaseExtractor
from vulnflow.loaders.base import BaseLoader

# =============================================================================
# _get_stage_name Tests
# =============================================================================


def test_get_stage_name_prefers_label():
    """_get_stage_name() prefers label over name."""

    class TestStage:
        name = "generic"
        label = "nvdcve-2.0-2023"

    obj = TestStage()
    assert PipelineExecutor._get_stage_name(obj) == "nvdcve-2.0-2023"


def test_get_stage_name_falls_back_to_name():
    """_get_stage_name() falls back to name when no label."""

    class TestStage:
        name = "custom_name"

    obj = TestStage()
    assert PipelineExecutor._get_stage_name(obj) == "custom_name"


def test_get_stage_name_fallback_to_class_name():
    """_get_stage_name() falls back to class name."""

    class TestStage:
        pass

    obj = TestStage()
    assert PipelineExecutor._get_stage_name(obj) == "TestStage"


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
    # label defaults to provenance_label: ClassName(path_basename)
    assert successful == ["MockExtractor(unknown)", "MockExtractor(unknown)"]
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


def test_run_extractors_concurrent_with_threads():
    """run_extractors() runs multiple extractors concurrently via threads."""
    call_times: list[float] = []

    class SlowExtractor(BaseExtractor):
        format = "slow"
        name = "SlowExtractor"

        def __init__(self, cve_id: str):
            super().__init__()
            self.cve_id = cve_id

        def extract(self) -> VulnDataset:
            time.sleep(0.2)
            call_times.append(time.time())
            ds = VulnDataset()
            ds._data = {"cve_id": np.array([self.cve_id])}
            return ds

    extractors = [SlowExtractor(f"CVE-{i}") for i in range(3)]

    start = time.time()
    dataset, successful, failed, errors = PipelineExecutor.run_extractors(extractors)
    elapsed = time.time() - start

    assert len(dataset) == 3
    assert len(successful) == 3
    assert errors == []
    # Should complete in ~0.2s (concurrent), not ~0.6s (sequential)
    assert elapsed < 0.5


# =============================================================================
# run_transforms Tests (phased execution)
# =============================================================================


@pytest.mark.asyncio
async def test_run_transforms_phases_respect_dependencies():
    """run_transforms() respects depends_on ordering via phases."""
    call_order: list[str] = []

    class TrackingAsync:
        def __init__(self, name: str, depends_on: list[str] | None = None):
            self.name = name
            self.depends_on = depends_on or []

        async def transform(self, dataset: VulnDataset) -> None:
            call_order.append(self.name)

    class TrackingSync:
        def __init__(self, name: str, depends_on: list[str] | None = None):
            self.name = name
            self.depends_on = depends_on or []

        def transform(self, dataset: VulnDataset) -> None:
            call_order.append(self.name)

    dataset = VulnDataset()
    transforms: list[Any] = [
        TrackingSync("priority", depends_on=["cvd_state"]),
        TrackingAsync("kev"),
        TrackingSync("cvd_state", depends_on=["kev"]),
    ]

    successful, failed, errors = await PipelineExecutor.run_transforms(
        dataset, transforms, strict=True
    )

    assert len(successful) == 3
    assert errors == []
    assert call_order.index("kev") < call_order.index("cvd_state")
    assert call_order.index("cvd_state") < call_order.index("priority")


@pytest.mark.asyncio
async def test_run_transforms_independent_in_same_phase():
    """Independent transforms in the same phase run concurrently."""
    call_times: dict[str, float] = {}

    class SlowAsync:
        def __init__(self, name: str):
            self.name = name
            self.depends_on: list[str] = []

        async def transform(self, dataset: VulnDataset) -> None:
            await asyncio.sleep(0.1)
            call_times[self.name] = time.time()

    dataset = VulnDataset()
    transforms: list[Any] = [SlowAsync("a"), SlowAsync("b"), SlowAsync("c")]

    start = time.time()
    successful, failed, errors = await PipelineExecutor.run_transforms(
        dataset, transforms, strict=True
    )
    elapsed = time.time() - start

    assert len(successful) == 3
    # Should complete in ~0.1s (concurrent within phase), not ~0.3s
    assert elapsed < 0.25


@pytest.mark.asyncio
async def test_run_transforms_strict_raises():
    """run_transforms() in strict mode raises on error."""

    class FailingTransform:
        name = "failing"
        depends_on: list[str] = []

        async def transform(self, dataset: VulnDataset) -> None:
            raise ValueError("Transform failed")

    dataset = VulnDataset()
    with pytest.raises(ValueError, match="Transform failed"):
        await PipelineExecutor.run_transforms(
            dataset, [FailingTransform()], strict=True
        )


@pytest.mark.asyncio
async def test_run_transforms_lenient_collects_errors():
    """run_transforms() in lenient mode collects errors and continues."""

    class FailingTransform:
        name = "failing"
        depends_on: list[str] = []

        async def transform(self, dataset: VulnDataset) -> None:
            raise ValueError("Transform failed")

    class GoodTransform:
        name = "good"
        depends_on: list[str] = []

        def transform(self, dataset: VulnDataset) -> None:
            pass

    dataset = VulnDataset()
    successful, failed, errors = await PipelineExecutor.run_transforms(
        dataset, [FailingTransform(), GoodTransform()], strict=False
    )

    assert "good" in successful
    assert "failing" in failed
    assert len(errors) == 1


@pytest.mark.asyncio
async def test_run_transforms_empty_list():
    """run_transforms() handles empty transform list."""
    dataset = VulnDataset()
    successful, failed, errors = await PipelineExecutor.run_transforms(
        dataset, [], strict=True
    )
    assert successful == []
    assert failed == []
    assert errors == []


@pytest.mark.asyncio
async def test_run_transforms_mixed_async_sync_same_phase():
    """Async and sync transforms in the same phase run concurrently."""
    call_times: dict[str, float] = {}

    class SlowAsync:
        name = "async_t"
        depends_on: list[str] = []

        async def transform(self, dataset: VulnDataset) -> None:
            import asyncio as _aio

            await _aio.sleep(0.1)
            call_times["async_t"] = time.time()

    class SlowSync:
        name = "sync_t"
        depends_on: list[str] = []

        def transform(self, dataset: VulnDataset) -> None:
            time.sleep(0.1)
            call_times["sync_t"] = time.time()

    dataset = VulnDataset()
    start = time.time()
    successful, failed, errors = await PipelineExecutor.run_transforms(
        dataset, [SlowAsync(), SlowSync()], strict=True
    )
    elapsed = time.time() - start

    assert set(successful) == {"async_t", "sync_t"}
    assert errors == []
    # Both in same phase, concurrent — should be ~0.1s not ~0.2s
    assert elapsed < 0.25


@pytest.mark.asyncio
async def test_run_transforms_sync_failure_through_to_thread():
    """Sync transform failure propagates correctly through asyncio.to_thread()."""
    from vulnflow.exceptions import TransformError

    class FailingSync:
        name = "failing_sync"
        depends_on: list[str] = []

        def transform(self, dataset: VulnDataset) -> None:
            raise TransformError("sync transform failed")

    dataset = VulnDataset()

    # Strict mode: error propagates
    with pytest.raises(TransformError, match="sync transform failed"):
        await PipelineExecutor.run_transforms(dataset, [FailingSync()], strict=True)

    # Lenient mode: error collected
    successful, failed, errors = await PipelineExecutor.run_transforms(
        dataset, [FailingSync()], strict=False
    )
    assert "failing_sync" in failed
    assert len(errors) == 1


def test_run_extractors_concurrent_failure_raises():
    """When 1 of N concurrent extractors fails, error propagates."""

    class GoodExtractor(BaseExtractor):
        format = "good"
        name = "GoodExtractor"

        def __init__(self, cve_id: str):
            super().__init__()
            self.cve_id = cve_id

        def extract(self) -> VulnDataset:
            time.sleep(0.05)
            ds = VulnDataset()
            ds._data = {"cve_id": np.array([self.cve_id])}
            return ds

    class FailingExtractor(BaseExtractor):
        format = "fail"
        name = "FailingExtractor"

        def extract(self) -> VulnDataset:
            raise RuntimeError("Extractor 2 failed")

    # 3 extractors, 1 fails — requires ThreadPool path (len > 1)
    extractors: list[BaseExtractor] = [
        GoodExtractor("CVE-1"),
        FailingExtractor(),
        GoodExtractor("CVE-3"),
    ]

    with pytest.raises(RuntimeError, match="Extractor 2 failed"):
        PipelineExecutor.run_extractors(extractors)


def test_run_extractors_populates_reporter_timings(nvd_sample_path: Path):
    """run_extractors() populates reporter.timings when reporter is provided."""
    from vulnflow.extractors.nvd import NVDExtractor
    from vulnflow.utils.progress import ProgressReporter

    ext = NVDExtractor(str(nvd_sample_path))

    os.environ["VULNFLOW_PROGRESS_MODE"] = "log"
    try:
        with ProgressReporter("Test", total=1, unit="feeds") as rpt:
            dataset, successful, failed, errors = PipelineExecutor.run_extractors(
                [ext], reporter=rpt
            )
    finally:
        os.environ.pop("VULNFLOW_PROGRESS_MODE", None)

    assert len(dataset) == 5
    assert len(rpt.timings) == 1
    stage_name = successful[0]
    assert stage_name in rpt.timings
    assert rpt.timings[stage_name] >= 0


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

    with pytest.raises(OSError, match="Loader failed"):
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


def test_run_loaders_populates_reporter_timings(tmp_path):
    """run_loaders() populates reporter.timings when reporter is provided."""
    from vulnflow.loaders.files import ParquetLoader
    from vulnflow.utils.progress import ProgressReporter

    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")
    loader = ParquetLoader(tmp_path / "out.parquet")

    os.environ["VULNFLOW_PROGRESS_MODE"] = "log"
    try:
        with ProgressReporter("Test", total=1, unit="loaders") as rpt:
            outputs, successful, failed, errors = PipelineExecutor.run_loaders(
                dataset, [loader], reporter=rpt
            )
    finally:
        os.environ.pop("VULNFLOW_PROGRESS_MODE", None)

    assert len(outputs) == 1
    assert len(rpt.timings) == 1
    stage_name = successful[0]
    assert stage_name in rpt.timings
    assert rpt.timings[stage_name] >= 0


# =============================================================================
# Stage Timing Detail Tests
# =============================================================================


def test_stage_detail_includes_timing_breakdown():
    """_stage_detail() returns detail and breakdown with sub-timings >= 0.1s."""

    class FakeStage:
        _timings = {"download": 10.5, "decompress": 1.2, "json": 0.04, "populate": 4.6}

    stage = FakeStage()
    detail, breakdown = PipelineExecutor._stage_detail(stage, "38,951 CVEs")

    assert detail == "38,951 CVEs"
    assert "download 10.5s" in breakdown
    assert "decompress 1.2s" in breakdown
    assert "populate 4.6s" in breakdown
    # json 0.04s is below threshold, should be omitted
    assert "json" not in breakdown


def test_stage_detail_no_timings():
    """_stage_detail() falls back to detail-only when no _timings."""

    class FakeStage:
        pass

    stage = FakeStage()
    detail, breakdown = PipelineExecutor._stage_detail(stage, "1,000 CVEs")

    assert detail == "1,000 CVEs"
    assert breakdown == ""


def test_stage_detail_all_sub_timings_below_threshold():
    """_stage_detail() omits breakdown when all sub-timings < 0.1s."""

    class FakeStage:
        _timings = {
            "download": 0.05,
            "decompress": 0.01,
            "json": 0.02,
            "populate": 0.03,
        }

    stage = FakeStage()
    detail, breakdown = PipelineExecutor._stage_detail(stage, "500 CVEs")

    assert detail == "500 CVEs"
    assert breakdown == ""


def test_nvd_extractor_timings_flow_through_executor(nvd_sample_path: Path):
    """NVDExtractor._timings flows through to executor detail string."""
    from vulnflow.extractors.nvd import NVDExtractor

    ext = NVDExtractor(str(nvd_sample_path))
    dataset, successful, failed, errors = PipelineExecutor.run_extractors([ext])

    assert len(dataset) == 5
    assert hasattr(ext, "_timings")
    assert "populate" in ext._timings


# =============================================================================
# Transform Timing Tests
# =============================================================================


@pytest.mark.asyncio
async def test_run_transforms_reports_individual_durations():
    """run_transforms() passes individual durations to reporter."""
    reported_durations: dict[str, float] = {}

    class MockReporter:
        def stage_started(self, name: str) -> None:
            pass

        def stage_completed(
            self,
            name: str,
            detail: str = "",
            duration: float | None = None,
            breakdown: str = "",
        ) -> None:
            reported_durations[name] = duration  # type: ignore[assignment]

    class SlowTransform:
        name = "slow"
        depends_on: list[str] = []

        async def transform(self, dataset: VulnDataset) -> None:
            await asyncio.sleep(0.1)

    class FastTransform:
        name = "fast"
        depends_on: list[str] = []

        async def transform(self, dataset: VulnDataset) -> None:
            pass

    dataset = VulnDataset()
    reporter = MockReporter()
    successful, failed, errors = await PipelineExecutor.run_transforms(
        dataset, [SlowTransform(), FastTransform()], strict=True, reporter=reporter  # type: ignore[arg-type]
    )

    assert "slow" in reported_durations
    assert "fast" in reported_durations
    assert reported_durations["slow"] is not None
    assert reported_durations["fast"] is not None
    # slow should take significantly longer than fast
    assert reported_durations["slow"] > reported_durations["fast"]


# =============================================================================
# Provenance Tests
# =============================================================================


class TestProvenanceInExecutor:
    """Tests for provenance tracking through executor stages."""

    def test_run_extractors_merges_provenance(self, nvd_sample_path: Path) -> None:
        """Provenance flows through run_extractors and merge."""
        from vulnflow.extractors.nvd import NVDExtractor

        ext = NVDExtractor(str(nvd_sample_path))
        merged, _, _, _ = PipelineExecutor.run_extractors([ext])
        assert any("NVDExtractor(" in e for e in merged.provenance["extractors"])
