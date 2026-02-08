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
                "mock_column", np.array(["A"] * len(dataset), dtype=object)
            )

    pipeline = VulnPipeline(
        extractors=[NVDExtractor("tests/fixtures/nvd_sample.json")],
        transformers=[MockTransform()],
    )

    result = pipeline.run()

    assert result.dataset is not None
    assert "mock_column" in result.dataset.columns
    assert "mock" in result.successful_stages  # Uses name attribute, not class name


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
    assert "kev" in result.dataset.columns
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
    warning_messages = [
        r.message for r in caplog.records if r.levelno == logging.WARNING
    ]
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
        "transformers": [
            {"type": "KEVEnricher", "source": "tests/fixtures/kev_sample.csv"}
        ],
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


def test_pipeline_error_is_vulnflow_error():
    """PipelineError inherits from VulnFlowError."""
    from vulnflow.exceptions import PipelineError, VulnFlowError

    err = PipelineError("test")
    assert isinstance(err, VulnFlowError)


def test_pipeline_multi_extract_transform_load(tmp_path):
    """Full pipeline: 2 extractors → phased transforms → 2 loaders (Flow 6)."""
    from vulnflow.extractors.nvd import NVDExtractor
    from vulnflow.loaders.files import JSONLoader, ParquetLoader
    from vulnflow.transforms.enrichers.kev import KEVEnricher
    from vulnflow.transforms.flatten import KEVFlattenTransform

    parquet_out = tmp_path / "archive.parquet"
    json_out = tmp_path / "archive.json"

    pipeline = VulnPipeline(
        extractors=[
            NVDExtractor("tests/fixtures/nvd_sample.json"),
            NVDExtractor("tests/fixtures/nvd_sample.json"),
        ],
        transformers=[
            KEVEnricher(source="tests/fixtures/kev_sample.csv"),
            KEVFlattenTransform(),
        ],
        loaders=[
            ParquetLoader(parquet_out),
            JSONLoader(json_out),
        ],
    )

    result = pipeline.run()

    # Extract: 2 extractors merge (same fixture → deduplicated by merge)
    assert result.dataset is not None
    assert len(result.dataset) > 0

    # Transform: phased — KEVEnricher (phase 0) → KEVFlattenTransform (phase 1)
    assert "kev" in result.dataset.columns
    assert "is_kev" in result.dataset.columns
    assert "KEVEnricher" in result.successful_stages
    assert "KEVFlattenTransform" in result.successful_stages

    # No errors
    assert result.errors == []

    # Load: both files written
    assert parquet_out.exists()
    assert json_out.exists()


@pytest.mark.asyncio
async def test_pipeline_run_from_async_context():
    """pipeline.run() works from within a running event loop (e.g. Jupyter)."""
    import asyncio

    from vulnflow.extractors.nvd import NVDExtractor

    # Verify we are inside a running event loop
    loop = asyncio.get_running_loop()
    assert loop is not None

    pipeline = VulnPipeline(
        extractors=[NVDExtractor("tests/fixtures/nvd_sample.json")],
    )

    # This would raise "asyncio.run() cannot be called from a running event loop"
    # without the nest_asyncio guard in pipeline.run()
    result = pipeline.run()

    assert result.dataset is not None
    assert len(result.dataset) > 0
    assert result.errors == []


