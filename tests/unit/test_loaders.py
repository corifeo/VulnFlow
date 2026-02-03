"""
Loaders - ParquetLoader and JSONLoader
Tests: Loader protocol, file writing

Layer: Load (ETL)
Dependencies: dataset
"""

from __future__ import annotations

from vulnflow.dataset import VulnDataset
from vulnflow.loaders.files import ParquetLoader


def test_parquet_loader_writes_file(tmp_path):
    """ParquetLoader.load() writes Parquet file."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    output_path = tmp_path / "output.parquet"
    loader = ParquetLoader(output_path)
    result = loader.load(dataset)

    assert output_path.exists()
    assert result["path"] == str(output_path)
    assert result["record_count"] == 3


def test_parquet_loader_in_pipeline(tmp_path):
    """Pipeline with ParquetLoader writes file."""
    from vulnflow.extractors.nvd import NVDExtractor
    from vulnflow.pipeline import VulnPipeline

    output_path = tmp_path / "pipeline_output.parquet"
    pipeline = VulnPipeline(
        extractors=[NVDExtractor("tests/fixtures/nvd_sample.json")],
        loaders=[ParquetLoader(output_path)],
    )

    result = pipeline.run()

    assert output_path.exists()
    assert len(result.dataset) == 3
    assert "parquet" in result.loader_outputs  # Uses name attribute, not class name
