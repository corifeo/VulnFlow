"""
Pipeline Core - VulnPipeline orchestration
Tests: PipelineResult, VulnPipeline composition

Layer: Core ETL orchestration
Dependencies: None (test fixtures only)
"""

from __future__ import annotations

import pytest

# Import to trigger registration decorators
import vulnflow.extractors.files  # noqa: F401
import vulnflow.extractors.nvd  # noqa: F401
import vulnflow.loaders.dataframe  # noqa: F401
import vulnflow.loaders.files  # noqa: F401
import vulnflow.transforms.enrichers.kev  # noqa: F401
from vulnflow.pipeline import PipelineResult, VulnPipeline


def test_pipeline_result_creation():
    """PipelineResult dataclass exists and has expected fields."""
    result = PipelineResult(
        dataset=None,
        loader_outputs={},
        errors=[],
        successful_stages=[],
        failed_stages=[],
        duration=0.0,
    )
    assert result.dataset is None
    assert result.loader_outputs == {}
    assert result.errors == []
    assert result.duration == 0.0


def test_pipeline_initialization():
    """VulnPipeline accepts extractors, transformers, loaders."""
    pipeline = VulnPipeline(extractors=[], transformers=[], loaders=[])
    assert pipeline.extractors == []
    assert pipeline.transformers == []
    assert pipeline.loaders == []


def test_pipeline_run_with_extractors_only():
    """Pipeline with just extractors returns dataset."""
    from vulnflow.extractors.nvd import NVDExtractor

    pipeline = VulnPipeline(extractors=[NVDExtractor("tests/fixtures/nvd_sample.json")])
    result = pipeline.run()

    assert isinstance(result, PipelineResult)
    assert result.dataset is not None
    assert len(result.dataset) > 0
    assert result.errors == []


def test_pipeline_composition_then():
    """Pipelines compose with .merge() - immutable."""
    from vulnflow.extractors.nvd import NVDExtractor
    from vulnflow.transforms.enrichers.kev import KEVEnricher

    p1 = VulnPipeline(extractors=[NVDExtractor("tests/fixtures/nvd_sample.json")])
    p2 = VulnPipeline(transformers=[KEVEnricher()])

    combined = p1.merge(p2)

    assert len(combined.extractors) == 1
    assert len(combined.transformers) == 1
    # Original pipelines unchanged (immutable)
    assert len(p1.transformers) == 0
    assert len(p2.extractors) == 0


def test_pipeline_composition_chain():
    """VulnPipeline.chain() combines multiple pipelines."""
    from vulnflow.extractors.nvd import NVDExtractor
    from vulnflow.transforms.enrichers.kev import KEVEnricher

    pipelines = [
        VulnPipeline(extractors=[NVDExtractor("tests/fixtures/nvd_sample.json")]),
        VulnPipeline(transformers=[KEVEnricher()]),
    ]

    combined = VulnPipeline.chain(pipelines)

    assert len(combined.extractors) == 1
    assert len(combined.transformers) == 1


def test_pipeline_exports():
    """VulnPipeline and PipelineResult exported from main package."""
    from vulnflow import PipelineResult, VulnPipeline

    assert VulnPipeline is not None
    assert PipelineResult is not None


def test_exception_exports():
    """Exception classes exported from main package."""
    from vulnflow import (
        EnricherError,
        ExtractorError,
        LoaderError,
        TransformError,
        VulnFlowError,
    )

    # All exceptions should inherit from VulnFlowError
    assert issubclass(ExtractorError, VulnFlowError)
    assert issubclass(EnricherError, VulnFlowError)
    assert issubclass(TransformError, VulnFlowError)
    assert issubclass(LoaderError, VulnFlowError)

    # VulnFlowError should inherit from Exception
    assert issubclass(VulnFlowError, Exception)


def test_pipeline_executes_transforms():
    """Pipeline executes transforms between extract and load."""
    import numpy as np

    from vulnflow.extractors.nvd import NVDExtractor

    # Mock transform that adds a column
    class MockTransform:
        name = "mock"
        depends_on: list[str] = []

        def transform(self, dataset):  # type: ignore[no-untyped-def]
            dataset.set_column(
                "mock_column", np.array(["A", "B", "C"], dtype=object)
            )

    pipeline = VulnPipeline(
        extractors=[NVDExtractor("tests/fixtures/nvd_sample.json")],
        transformers=[MockTransform()],
    )

    result = pipeline.run()

    assert result.dataset is not None
    assert "mock_column" in result.dataset.columns
    assert "mock" in result.successful_stages  # Uses name attribute, not class name


def test_pipeline_10_pipelines_with_10_stages():
    """Pipeline composition scales to 10 pipelines x 10 stages each."""
    from tests.conftest import MockExtractor, MockLoader, MockTransformer

    # Create 10 pipelines, each with 10 stages (mix of extractors, transformers, loaders)
    pipelines = []
    all_extractors: list[MockExtractor] = []
    all_transformers: list[MockTransformer] = []
    all_loaders: list[MockLoader] = []

    for i in range(10):
        # Each pipeline gets a unique mix of stages
        extractors = [
            MockExtractor(name=f"extractor_{i}_{j}", num_records=2) for j in range(3)
        ]
        transformers = [MockTransformer(name=f"transformer_{i}_{j}") for j in range(4)]
        loaders = [MockLoader(name=f"loader_{i}_{j}") for j in range(3)]

        all_extractors.extend(extractors)
        all_transformers.extend(transformers)
        all_loaders.extend(loaders)

        pipelines.append(
            VulnPipeline(
                extractors=extractors,
                transformers=transformers,
                loaders=loaders,
            )
        )

    # Chain all 10 pipelines
    combined = VulnPipeline.chain(pipelines)

    # Verify composition integrity
    assert len(combined.extractors) == 30  # 10 pipelines x 3 extractors
    assert len(combined.transformers) == 40  # 10 pipelines x 4 transformers
    assert len(combined.loaders) == 30  # 10 pipelines x 3 loaders

    # Verify all original mock objects are preserved (not copied)
    for extractor in all_extractors:
        assert extractor in combined.extractors
    for transformer in all_transformers:
        assert transformer in combined.transformers
    for loader in all_loaders:
        assert loader in combined.loaders

    # Run the pipeline and verify extraction works
    result = combined.run()

    assert isinstance(result, PipelineResult)
    assert result.dataset is not None
    # 30 extractors x 2 records each = 60 total records
    assert len(result.dataset) == 60
    assert result.errors == []
    # 30 extractors + 40 transformers + 30 loaders = 100 successful stages
    assert len(result.successful_stages) == 100

    # Verify all extractors were called
    for extractor in all_extractors:
        assert extractor.extract_called, f"{extractor.name} was not called"

    # Verify all loaders were called
    for loader in all_loaders:
        assert loader.load_called, f"{loader.name} was not called"


def test_pipeline_with_enrichers():
    """Pipeline runs async enrichers."""
    from vulnflow.extractors.nvd import NVDExtractor
    from vulnflow.transforms.enrichers.kev import KEVEnricher

    pipeline = VulnPipeline(
        extractors=[NVDExtractor("tests/fixtures/nvd_sample.json")],
        transformers=[KEVEnricher(source="tests/fixtures/kev_sample.csv")],
    )

    result = pipeline.run()

    assert result.dataset is not None
    assert "kev_date_added" in result.dataset.columns
    assert "KEVEnricher" in result.successful_stages


def test_pipeline_strict_mode_raises_on_enricher_error():
    """Strict mode raises on enricher failure."""
    from fixtures.broken_enricher import BrokenEnricher
    from vulnflow.exceptions import EnricherError
    from vulnflow.extractors.nvd import NVDExtractor

    pipeline = VulnPipeline(
        extractors=[NVDExtractor("tests/fixtures/nvd_sample.json")],
        transformers=[BrokenEnricher()],
    )

    with pytest.raises(EnricherError, match="simulated failure"):
        pipeline.run(strict=True)


def test_pipeline_strict_mode_raises_on_transform_error():
    """Strict mode raises on transform failure."""
    from fixtures.broken_enricher import BrokenTransform
    from vulnflow.exceptions import TransformError
    from vulnflow.extractors.nvd import NVDExtractor

    pipeline = VulnPipeline(
        extractors=[NVDExtractor("tests/fixtures/nvd_sample.json")],
        transformers=[BrokenTransform()],
    )

    with pytest.raises(TransformError, match="simulated failure"):
        pipeline.run(strict=True)


def test_pipeline_lenient_mode_continues_on_error():
    """Lenient mode collects errors, continues execution."""
    from fixtures.broken_enricher import BrokenEnricher
    from vulnflow.extractors.nvd import NVDExtractor
    from vulnflow.transforms.enrichers.kev import KEVEnricher

    pipeline = VulnPipeline(
        extractors=[NVDExtractor("tests/fixtures/nvd_sample.json")],
        transformers=[
            KEVEnricher(source="tests/fixtures/kev_sample.csv"),
            BrokenEnricher(),
        ],
    )

    # Should NOT raise
    result = pipeline.run(strict=False)

    assert len(result.errors) == 1
    assert "BrokenEnricher" in result.failed_stages
    assert "KEVEnricher" in result.successful_stages
    assert result.dataset is not None


def test_pipeline_lenient_mode_is_default():
    """Lenient mode is default (strict=False)."""
    from fixtures.broken_enricher import BrokenEnricher
    from vulnflow.extractors.nvd import NVDExtractor

    pipeline = VulnPipeline(
        extractors=[NVDExtractor("tests/fixtures/nvd_sample.json")],
        transformers=[BrokenEnricher()],
    )

    # Should NOT raise (default is lenient)
    result = pipeline.run()

    assert len(result.errors) == 1
    assert "BrokenEnricher" in result.failed_stages


def test_pipeline_extractor_failure_always_raises():
    """Extractor failures always raise, even in lenient mode."""
    from vulnflow.exceptions import ExtractorError

    class BrokenExtractor:
        def extract(self) -> list:
            raise ExtractorError("Broken extractor")

    pipeline = VulnPipeline(extractors=[BrokenExtractor()])

    with pytest.raises(ExtractorError, match="Broken extractor"):
        pipeline.run(strict=False)


def test_pipeline_loader_failure_always_raises():
    """Loader failures always raise, even in lenient mode."""
    from vulnflow.exceptions import LoaderError
    from vulnflow.extractors.nvd import NVDExtractor

    class BrokenLoader:
        def load(self, dataset: object) -> None:
            raise LoaderError("Broken loader")

    pipeline = VulnPipeline(
        extractors=[NVDExtractor("tests/fixtures/nvd_sample.json")],
        loaders=[BrokenLoader()],
    )

    with pytest.raises(LoaderError, match="Broken loader"):
        pipeline.run(strict=False)


def test_pipeline_runs_enrichers_concurrently():
    """Pipeline runs multiple enrichers concurrently with asyncio.gather()."""
    import asyncio
    import time

    from vulnflow.extractors.nvd import NVDExtractor

    call_order: list[str] = []

    class SlowEnricher1:
        name = "slow1"
        depends_on: list[str] = []

        async def transform(self, dataset: object) -> None:
            call_order.append("slow1_start")
            await asyncio.sleep(0.1)
            call_order.append("slow1_end")

    class SlowEnricher2:
        name = "slow2"
        depends_on: list[str] = []

        async def transform(self, dataset: object) -> None:
            call_order.append("slow2_start")
            await asyncio.sleep(0.1)
            call_order.append("slow2_end")

    pipeline = VulnPipeline(
        extractors=[NVDExtractor("tests/fixtures/nvd_sample.json")],
        transformers=[SlowEnricher1(), SlowEnricher2()],
        loaders=[],
    )

    start = time.time()
    pipeline.run()
    elapsed = time.time() - start

    # If running concurrently, should take ~0.1s not ~0.2s
    assert elapsed < 0.15, f"Enrichers ran sequentially (took {elapsed:.2f}s)"

    # Both should start before either ends (concurrent)
    assert call_order.index("slow1_start") < call_order.index("slow1_end")
    assert call_order.index("slow2_start") < call_order.index("slow2_end")


def test_pipeline_lenient_mode_logs_warnings(caplog: pytest.LogCaptureFixture):
    """Pipeline logs warnings before collecting errors in lenient mode."""
    import logging

    from fixtures.broken_enricher import BrokenEnricher
    from vulnflow.extractors.nvd import NVDExtractor

    pipeline = VulnPipeline(
        extractors=[NVDExtractor("tests/fixtures/nvd_sample.json")],
        transformers=[BrokenEnricher()],
        loaders=[],
    )

    with caplog.at_level(logging.WARNING):
        pipeline.run(strict=False)

    # Should have logged a warning about the failure
    assert len(caplog.records) >= 1, "Expected at least one log record"
    warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any(
        "BrokenEnricher" in msg or "broken" in msg.lower() for msg in warning_messages
    ), f"Expected warning about BrokenEnricher, got: {warning_messages}"


def test_pipeline_then_chains_transforms_only():
    """then() chains transform-only pipelines (sklearn-like behavior)."""
    from tests.conftest import MockTransformer

    t1 = MockTransformer(name="transform1")
    t2 = MockTransformer(name="transform2")
    t3 = MockTransformer(name="transform3")

    p1 = VulnPipeline(transformers=[t1, t2])
    p2 = VulnPipeline(transformers=[t3])

    chained = p1.then(p2)

    # Should concatenate transformers
    assert len(chained.transformers) == 3
    assert chained.transformers[0] is t1
    assert chained.transformers[1] is t2
    assert chained.transformers[2] is t3

    # Should have no extractors or loaders
    assert len(chained.extractors) == 0
    assert len(chained.loaders) == 0

    # Original pipelines unchanged (immutable)
    assert len(p1.transformers) == 2
    assert len(p2.transformers) == 1


def test_pipeline_then_rejects_extractors():
    """then() raises ValueError if extractors present."""
    from tests.conftest import MockExtractor, MockTransformer

    p1 = VulnPipeline(extractors=[MockExtractor()])
    p2 = VulnPipeline(transformers=[MockTransformer()])

    with pytest.raises(
        ValueError,
        match="then\\(\\) requires transform-only pipelines but self has extractors",
    ):
        p1.then(p2)


def test_pipeline_then_rejects_loaders():
    """then() raises ValueError if loaders present."""
    from tests.conftest import MockLoader, MockTransformer

    p1 = VulnPipeline(transformers=[MockTransformer()])
    p2 = VulnPipeline(loaders=[MockLoader()])

    with pytest.raises(
        ValueError,
        match="then\\(\\) requires transform-only pipelines but other has loaders",
    ):
        p1.then(p2)


def test_pipeline_then_empty_pipelines():
    """then() handles empty pipelines correctly."""
    p1 = VulnPipeline()
    p2 = VulnPipeline()

    chained = p1.then(p2)

    # Should create empty pipeline (no stages)
    assert len(chained.extractors) == 0
    assert len(chained.transformers) == 0
    assert len(chained.loaders) == 0


def test_pipeline_from_config_basic():
    """Build pipeline from dict config."""
    config = {
        "extractors": [{"type": "nvd_json", "path": "tests/fixtures/nvd_sample.json"}],
        "transformers": [{"type": "KEVEnricher", "source": "tests/fixtures/kev_sample.csv"}],
        "loaders": [{"type": "json", "path": "out.json"}],
    }

    pipeline = VulnPipeline.from_config(config)

    assert len(pipeline.extractors) == 1
    assert len(pipeline.transformers) == 1
    assert len(pipeline.loaders) == 1


def test_pipeline_from_config_string_names():
    """Accept string names for default constructors."""
    config = {
        "extractors": [],
        "transformers": ["KEVEnricher"],
        "loaders": [],
    }

    pipeline = VulnPipeline.from_config(config)

    assert len(pipeline.extractors) == 0
    assert len(pipeline.transformers) == 1
    assert len(pipeline.loaders) == 0


def test_pipeline_from_config_with_params():
    """Pass params to constructors."""
    config = {
        "extractors": [{"type": "parquet", "path": "data.parquet"}],
        "transformers": [],
        "loaders": [{"type": "parquet", "path": "out.parquet"}],
    }

    pipeline = VulnPipeline.from_config(config)

    assert len(pipeline.extractors) == 1
    assert len(pipeline.loaders) == 1


def test_pipeline_from_config_empty():
    """Handle empty config."""
    config = {
        "extractors": [],
        "transformers": [],
        "loaders": [],
    }

    pipeline = VulnPipeline.from_config(config)

    assert len(pipeline.extractors) == 0
    assert len(pipeline.transformers) == 0
    assert len(pipeline.loaders) == 0


def test_pipeline_from_config_unknown_extractor():
    """Raise KeyError for unknown type."""
    config = {
        "extractors": [{"type": "unknown_extractor", "path": "data.json"}],
        "transformers": [],
        "loaders": [],
    }

    with pytest.raises(KeyError, match="Extractor not found: unknown_extractor"):
        VulnPipeline.from_config(config)
