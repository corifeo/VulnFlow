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


