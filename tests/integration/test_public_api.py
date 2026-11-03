"""
Public API Integration Tests
Tests: Exported classes and functions work end-to-end

Layer: Integration testing
"""

from __future__ import annotations

import importlib
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "module, names",
    [
        (
            "vulnflow.extractors",
            [
                "NVDExtractor",
                "NVDAPIExtractor",
                "MSRCExtractor",
                "ParquetExtractor",
                "JSONExtractor",
                "PandasExtractor",
                "PolarsExtractor",
            ],
        ),
        (
            "vulnflow.loaders",
            ["ParquetLoader", "JSONLoader", "PandasLoader", "PolarsLoader"],
        ),
    ],
)
def test_subpackages_export_concrete_stages(module: str, names: list[str]) -> None:
    mod = importlib.import_module(module)
    missing = [n for n in names if n not in mod.__all__ or not hasattr(mod, n)]
    assert missing == []


@pytest.mark.parametrize(
    "module",
    [
        "vulnflow.extractors",
        "vulnflow.loaders",
        "vulnflow.dataset",
        "vulnflow.extractors.nvd",
        "vulnflow.loaders.files",
        "vulnflow.loaders.dataframe",
    ],
)
def test_each_module_imports_first_in_a_fresh_interpreter(module: str) -> None:
    """Catches circular imports that a shared test process would hide."""
    subprocess.run([sys.executable, "-c", f"import {module}"], check=True)


def test_version_comes_from_pyproject() -> None:
    """Fails on the first version bump if __version__ is ever hard-coded again."""
    import tomllib
    from pathlib import Path

    import vulnflow

    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf8"))
    assert vulnflow.__version__ == project["project"]["version"]


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
    assert len(dataset) == 5

    # Export to Parquet (file loaders are lossless)
    output_path = tmp_path / "output.parquet"
    dataset.to_parquet(output_path)

    # Reload
    dataset2 = VulnDataset.from_parquet(output_path)
    assert len(dataset2) == 5


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
    assert len(result.dataset) == 5
    assert output_path.exists()
    assert "ParquetLoader(pipeline.parquet)" in result.loader_outputs
