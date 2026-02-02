"""
Pipeline Core - VulnPipeline orchestration
Tests: PipelineResult, VulnPipeline composition

Layer: Core ETL orchestration
Dependencies: None (test fixtures only)
"""

from __future__ import annotations

from vulnflow.pipeline import PipelineResult, VulnPipeline


def test_pipeline_result_creation():
    """PipelineResult dataclass exists and has expected fields."""
    result = PipelineResult(
        dataset=None,
        loader_outputs={},
        errors=[],
        successful_stages=[],
        failed_stages=[],
        duration=0.0
    )
    assert result.dataset is None
    assert result.loader_outputs == {}
    assert result.errors == []
    assert result.duration == 0.0


def test_pipeline_initialization():
    """VulnPipeline accepts extractors, transformers, loaders."""
    pipeline = VulnPipeline(
        extractors=[],
        transformers=[],
        loaders=[]
    )
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
    """Pipelines compose with .then() - immutable."""
    from vulnflow.extractors.nvd import NVDExtractor
    from vulnflow.transforms.enrichers.kev import KEVEnricher

    p1 = VulnPipeline(extractors=[NVDExtractor("tests/fixtures/nvd_sample.json")])
    p2 = VulnPipeline(transformers=[KEVEnricher()])

    combined = p1.then(p2)

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
        extractors = [MockExtractor(name=f"extractor_{i}_{j}", num_records=2) for j in range(3)]
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
    assert len(result.successful_stages) == 30  # All extractors succeeded

    # Verify all extractors were called
    for extractor in all_extractors:
        assert extractor.extract_called, f"{extractor.name} was not called"
