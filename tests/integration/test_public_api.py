"""
Public API Integration Tests
Tests: Exported classes and functions work end-to-end

Layer: Integration testing
"""

from __future__ import annotations


def test_public_api_exports():
    """Public API exports all required classes."""
    from vulnflow import (
        BaseExtractor,
        BaseLoader,
        Extractor,
        Loader,
        PipelineResult,
        VulnDataset,
        VulnPipeline,
        VulnRecord,
    )

    assert VulnDataset is not None
    assert VulnRecord is not None
    assert VulnPipeline is not None
    assert PipelineResult is not None
    assert Extractor is not None
    assert BaseExtractor is not None
    assert Loader is not None
    assert BaseLoader is not None


def test_public_api_end_to_end(tmp_path):
    """End-to-end usage via public API."""
    from vulnflow import VulnDataset

    # Load from NVD
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")
    assert len(dataset) == 3

    # Export to Parquet
    output_path = tmp_path / "output.parquet"
    dataset.to_parquet(output_path)

    # Reload
    dataset2 = VulnDataset.from_parquet(output_path)
    assert len(dataset2) == 3


def test_public_api_pipeline_end_to_end(tmp_path):
    """Full pipeline via public API."""
    from vulnflow import VulnPipeline
    from vulnflow.extractors.nvd import NVDExtractor
    from vulnflow.loaders.files import ParquetLoader

    output_path = tmp_path / "pipeline.parquet"
    pipeline = VulnPipeline(
        extractors=[NVDExtractor("tests/fixtures/nvd_sample.json")],
        loaders=[ParquetLoader(output_path)],
    )

    result = pipeline.run()

    assert result.dataset is not None
    assert len(result.dataset) == 3
    assert output_path.exists()
    assert "parquet" in result.loader_outputs  # Uses name attribute, not class name


def test_default_transforms_registered():
    """All default transforms registered (KEV, EPSS, EventInference, CVDState)."""
    # Import transforms to trigger registration decorators
    import vulnflow.transforms.computed  # noqa: F401
    import vulnflow.transforms.enrichers  # noqa: F401
    import vulnflow.transforms.inferences  # noqa: F401
    from vulnflow.registry import Registry

    defaults = Registry.get_default_transforms()

    # Should have at least 4 default transforms
    # (CVDDesiderataTransform was consolidated into CVDStateTransform)
    assert len(defaults) >= 4, f"Expected at least 4 defaults, got {len(defaults)}"

    transform_names = [t.__class__.__name__ for t in defaults]

    # Enrichers (async I/O-bound)
    assert "KEVEnricher" in transform_names
    assert "EPSSEnricher" in transform_names

    # Inferences (sync with confidence)
    assert "EventInference" in transform_names

    # Computed (sync pure computation)
    # CVDStateTransform now computes all 19 CVD fields (including desiderata)
    assert "CVDStateTransform" in transform_names


def test_default_transforms_instantiable():
    """Default transforms can be instantiated without arguments."""
    from vulnflow.registry import Registry

    defaults = Registry.get_default_transforms()

    # All defaults should be instances, not classes
    for transform in defaults:
        assert not isinstance(
            transform, type
        ), f"{transform} should be an instance, not a class"
