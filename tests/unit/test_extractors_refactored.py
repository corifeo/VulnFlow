"""
Tests for refactored extractors returning VulnDataset.

Verifies:
- Extractors return VulnDataset (not list[VulnRecord])
- NVDExtractor owns conversion logic
- ParquetExtractor implements direct load
- Flatten transforms produce scalar columns (flattening moved from extractor)
- BaseExtractor format validation

Layer: Test
Dependencies: extractors, dataset
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from vulnflow.dataset import VulnDataset
from vulnflow.exceptions import ExtractorError
from vulnflow.extractors.base import BaseExtractor
from vulnflow.extractors.files import ParquetExtractor
from vulnflow.extractors.nvd import NVDExtractor


def test_nvd_extractor_returns_dataset() -> None:
    """NVDExtractor.extract() returns VulnDataset, not list."""
    extractor = NVDExtractor("tests/fixtures/nvd_sample.json")
    result = extractor.extract()

    assert isinstance(result, VulnDataset)
    assert len(result) == 3
    assert "cve_id" in result.columns


def test_parquet_extractor_returns_dataset() -> None:
    """ParquetExtractor.extract() returns VulnDataset."""
    # First create a parquet file
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    with tempfile.TemporaryDirectory() as tmpdir:
        parquet_path = Path(tmpdir) / "test.parquet"
        dataset.to_parquet(str(parquet_path))

        # Extract it
        extractor = ParquetExtractor(str(parquet_path))
        result = extractor.extract()

        assert isinstance(result, VulnDataset)
        assert len(result) == 3


def test_cvss_flatten_transform_produces_columns() -> None:
    """CVSSFlattenTransform produces flattened CVSS columns from nested array."""
    from vulnflow.transforms.flatten import CVSSFlattenTransform

    extractor = NVDExtractor("tests/fixtures/nvd_sample.json")
    dataset = extractor.extract()

    # Before transform: no flattened columns
    assert "cvss_base_score" not in dataset.columns

    # Apply flatten transform
    CVSSFlattenTransform().transform(dataset)

    # After transform: flattened columns exist
    assert "cvss_base_score" in dataset.columns
    assert "cvss_base_severity" in dataset.columns
    assert "cvss_version" in dataset.columns
    assert "cvss_vector_string" in dataset.columns
    assert "cvss_source" in dataset.columns


def test_cwe_flatten_transform_produces_columns() -> None:
    """CWEPrimaryTransform produces flattened CWE columns from nested array."""
    from vulnflow.transforms.flatten import CWEPrimaryTransform

    extractor = NVDExtractor("tests/fixtures/nvd_sample.json")
    dataset = extractor.extract()

    # Before transform: no flattened columns
    assert "cwe_id" not in dataset.columns

    # Apply flatten transform
    CWEPrimaryTransform().transform(dataset)

    # After transform: flattened columns exist
    assert "cwe_id" in dataset.columns
    assert "cwe_name" in dataset.columns


def test_cpe_flatten_transform_produces_columns() -> None:
    """CPEPrimaryTransform produces flattened CPE columns from nested array."""
    from vulnflow.transforms.flatten import CPEPrimaryTransform

    extractor = NVDExtractor("tests/fixtures/nvd_sample.json")
    dataset = extractor.extract()

    # Before transform: no flattened columns
    assert "cpe_vendor" not in dataset.columns

    # Apply flatten transform
    CPEPrimaryTransform().transform(dataset)

    # After transform: flattened columns exist
    assert "cpe_vendor" in dataset.columns
    assert "cpe_product" in dataset.columns
    assert "cpe_version" in dataset.columns


def test_cvss_flatten_selects_authoritative() -> None:
    """CVSSFlattenTransform selects is_authoritative=True entries."""

    from vulnflow.transforms.flatten import CVSSFlattenTransform

    extractor = NVDExtractor("tests/fixtures/nvd_sample.json")
    dataset = extractor.extract()

    # Apply flatten transform
    CVSSFlattenTransform().transform(dataset)

    # Check that CVSS sources are properly set (authoritative = nvd)
    cvss_sources = dataset.get_column("cvss_source")
    # All should be "nvd" or None (if no CVSS)
    for source in cvss_sources:
        assert source in ("nvd", None), f"Unexpected source: {source}"


def test_parquet_extractor_direct_load() -> None:
    """ParquetExtractor loads directly without round-trip conversion."""
    # Create a dataset with known data
    original = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    with tempfile.TemporaryDirectory() as tmpdir:
        parquet_path = Path(tmpdir) / "test.parquet"
        original.to_parquet(str(parquet_path))

        # Extract it directly
        extractor = ParquetExtractor(str(parquet_path))
        loaded = extractor.extract()

        # Should have same columns and length
        assert set(loaded.columns) == set(original.columns)
        assert len(loaded) == len(original)


def test_parquet_extractor_file_not_found() -> None:
    """ParquetExtractor raises FileNotFoundError for missing file."""
    extractor = ParquetExtractor("nonexistent.parquet")

    with pytest.raises(FileNotFoundError):
        extractor.extract()


def test_nvd_extractor_encodes_complex_fields() -> None:
    """NVDExtractor encodes complex fields (lists) as JSON strings."""
    extractor = NVDExtractor("tests/fixtures/nvd_sample.json")
    dataset = extractor.extract()

    # Complex fields should be JSON strings
    assert "cvss_scores" in dataset.columns
    assert "cwes" in dataset.columns
    assert "cpes" in dataset.columns
    assert "references" in dataset.columns

    # Sample one value to ensure it's a JSON string
    cvss_scores = dataset.get_column("cvss_scores")
    if cvss_scores[0] is not None:
        # Should be a string containing JSON
        assert isinstance(cvss_scores[0], str)


# BaseExtractor format validation tests


def test_base_extractor_validates_format_attribute() -> None:
    """BaseExtractor raises ExtractorError if format not set."""

    class MissingFormatExtractor(BaseExtractor):
        # Intentionally missing format attribute
        name = "MissingFormatExtractor"

        def extract(self) -> VulnDataset:
            return VulnDataset()

    with pytest.raises(
        ExtractorError, match="must define a non-empty 'format' class attribute"
    ):
        MissingFormatExtractor()


def test_base_extractor_accepts_valid_format() -> None:
    """BaseExtractor does not raise if format is set."""

    class ValidFormatExtractor(BaseExtractor):
        format = "json"
        name = "ValidFormatExtractor"

        def extract(self) -> VulnDataset:
            return VulnDataset()

    # Should not raise
    extractor = ValidFormatExtractor()
    assert extractor.format == "json"


def test_base_extractor_rejects_empty_format() -> None:
    """BaseExtractor raises ExtractorError if format is empty string."""

    class EmptyFormatExtractor(BaseExtractor):
        format = ""
        name = "EmptyFormatExtractor"

        def extract(self) -> VulnDataset:
            return VulnDataset()

    with pytest.raises(
        ExtractorError, match="must define a non-empty 'format' class attribute"
    ):
        EmptyFormatExtractor()
