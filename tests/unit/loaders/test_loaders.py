"""
Loaders - ParquetLoader and JSONLoader
Tests: Loader protocol, file writing

Layer: Load (ETL)
Dependencies: dataset
"""

from __future__ import annotations

import pytest

from vulnflow.dataset import VulnDataset
from vulnflow.exceptions import LoaderError
from vulnflow.loaders.base import BaseLoader
from vulnflow.loaders.files import ParquetLoader


def test_parquet_loader_writes_file(tmp_path):
    """ParquetLoader.load() writes Parquet file."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    output_path = tmp_path / "output.parquet"
    loader = ParquetLoader(output_path)
    result = loader.load(dataset)

    assert output_path.exists()
    assert result["path"] == str(output_path)
    assert result["record_count"] == 5


def test_parquet_loader_handles_mixed_datetime_string(tmp_path):
    """ParquetLoader handles object arrays with mixed datetime and string values."""
    from datetime import UTC, datetime

    import numpy as np

    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    # Simulate what EventInference produces: mix of datetime objects and ISO strings
    mixed = np.array(
        [
            datetime(2024, 1, 15, tzinfo=UTC),
            "2024-02-20",
            None,
            datetime(2024, 3, 10, tzinfo=UTC),
            "2024-04-05T12:00:00+00:00",
        ],
        dtype=object,
    )
    dataset.set_column("event_v_at", mixed)

    output_path = tmp_path / "mixed.parquet"
    loader = ParquetLoader(output_path)
    result = loader.load(dataset)

    assert output_path.exists()
    assert result["record_count"] == 5


def test_parquet_loader_handles_string_datetime_column(tmp_path):
    """ParquetLoader converts string-only datetime columns by name suffix."""
    import numpy as np

    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    # Simulate KEVFlattenTransform: all strings in a datetime-named column
    dates = np.array(
        ["2024-01-15", "2024-02-20", None, "2024-03-10", "2024-04-05"],
        dtype=object,
    )
    dataset.set_column("kev_date_added", dates)

    output_path = tmp_path / "string_dates.parquet"
    loader = ParquetLoader(output_path)
    result = loader.load(dataset)

    assert output_path.exists()
    assert result["record_count"] == 5


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
    assert len(result.dataset) == 5
    assert any("ParquetLoader" in k for k in result.loader_outputs)  # Uses label property


def test_base_loader_validates_format_attribute() -> None:
    """BaseLoader raises LoaderError if format not set."""

    class MissingFormatLoader(BaseLoader):
        # Intentionally missing format attribute
        name = "MissingFormatLoader"

        def load(self, dataset: VulnDataset) -> dict[str, any]:
            return {}

    with pytest.raises(
        LoaderError, match="must define a non-empty 'format' class attribute"
    ):
        MissingFormatLoader()


def test_base_loader_accepts_valid_format() -> None:
    """BaseLoader does not raise if format is set."""

    class ValidFormatLoader(BaseLoader):
        format = "json"
        name = "ValidFormatLoader"

        def load(self, dataset: VulnDataset) -> dict[str, any]:
            return {}

    # Should not raise
    loader = ValidFormatLoader()
    assert loader.format == "json"


def test_base_loader_rejects_empty_format() -> None:
    """BaseLoader raises LoaderError if format is empty string."""

    class EmptyFormatLoader(BaseLoader):
        format = ""
        name = "EmptyFormatLoader"

        def load(self, dataset: VulnDataset) -> dict[str, any]:
            return {}

    with pytest.raises(
        LoaderError, match="must define a non-empty 'format' class attribute"
    ):
        EmptyFormatLoader()


def test_all_loaders_have_format_attribute() -> None:
    """All built-in loaders have correct format attribute."""
    from vulnflow.loaders.dataframe import PandasLoader, PolarsLoader
    from vulnflow.loaders.files import JSONLoader, ParquetLoader

    # Verify each loader has format attribute with correct value
    loaders = [
        (ParquetLoader, "parquet"),
        (JSONLoader, "json"),
        (PandasLoader, "pandas"),
        (PolarsLoader, "polars"),
    ]

    for loader_class, expected_format in loaders:
        assert hasattr(
            loader_class, "format"
        ), f"{loader_class.__name__} missing format attribute"
        assert (
            loader_class.format == expected_format
        ), f"{loader_class.__name__}.format = {loader_class.format}, expected {expected_format}"
