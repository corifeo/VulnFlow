"""
Extractors - NVDExtractor, ParquetExtractor, and base protocol
Tests: NVDExtractor parsing, ParquetExtractor, Extractor protocol, flatten transforms

Layer: Extract (ETL)
Dependencies: schema
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from vulnflow.dataset import VulnDataset
from vulnflow.exceptions import ExtractorError
from vulnflow.extractors.base import BaseExtractor
from vulnflow.extractors.files import ParquetExtractor
from vulnflow.extractors.nvd import NVDExtractor


def test_nvd_extractor_returns_dataset():
    """NVDExtractor.extract() returns VulnDataset."""
    extractor = NVDExtractor("tests/fixtures/nvd_sample.json")
    dataset = extractor.extract()

    assert isinstance(dataset, VulnDataset)
    assert len(dataset) == 5
    assert "cve_id" in dataset.columns


def test_nvd_extractor_parses_cve_id():
    """NVDExtractor correctly parses CVE IDs."""
    extractor = NVDExtractor("tests/fixtures/nvd_sample.json")
    dataset = extractor.extract()

    cve_ids = dataset.get_column("cve_id")
    assert cve_ids[0] == "CVE-2024-0001"
    assert cve_ids[1] == "CVE-2024-0002"
    assert cve_ids[2] == "CVE-2024-0003"


def test_nvd_extractor_parses_nvd_timestamps():
    """NVDExtractor populates nvd_published_at and nvd_modified_at."""
    extractor = NVDExtractor("tests/fixtures/nvd_sample.json")
    dataset = extractor.extract()

    # First record has timestamps
    published = dataset.get_column("nvd_published_at")
    assert published[0] is not None
    assert published[0].year == 2024

    modified = dataset.get_column("nvd_modified_at")
    assert modified[0] is not None


def test_nvd_extractor_parses_cvss():
    """NVDExtractor parses CVSS v3 scores into cvss_scores array."""
    extractor = NVDExtractor("tests/fixtures/nvd_sample.json")
    dataset = extractor.extract()

    # First record has CVSS in nested array
    cvss_raw = dataset.get_column("cvss_scores")[0]
    cvss_data = json.loads(cvss_raw)
    assert len(cvss_data) >= 1
    assert cvss_data[0]["base_score"] == 9.8
    assert cvss_data[0]["base_severity"] == "CRITICAL"
    assert cvss_data[0]["source"] == "nvd"

    # Second record (CVE-2024-0002) has no CVSS metrics
    cvss_raw_2 = dataset.get_column("cvss_scores")[1]
    cvss_data_2 = json.loads(cvss_raw_2)
    assert cvss_data_2 == []


def test_nvd_extractor_parses_cwes():
    """NVDExtractor parses CWE weaknesses into cwes array."""
    extractor = NVDExtractor("tests/fixtures/nvd_sample.json")
    dataset = extractor.extract()

    # First record has CWE-94 in nested array
    cwes_raw = dataset.get_column("cwes")[0]
    cwes_data = json.loads(cwes_raw)
    assert len(cwes_data) >= 1
    assert cwes_data[0]["id"] == "CWE-94"

    # Second record has no CWEs
    cwes_raw_2 = dataset.get_column("cwes")[1]
    cwes_data_2 = json.loads(cwes_raw_2)
    assert cwes_data_2 == []

    # Third record has CWE-79
    cwes_raw_3 = dataset.get_column("cwes")[2]
    cwes_data_3 = json.loads(cwes_raw_3)
    assert cwes_data_3[0]["id"] == "CWE-79"


def test_nvd_extractor_parses_cpes():
    """NVDExtractor parses CPE configurations into cpes array."""
    extractor = NVDExtractor("tests/fixtures/nvd_sample.json")
    dataset = extractor.extract()

    # First record has CPE for example:product in nested array
    cpes_raw = dataset.get_column("cpes")[0]
    cpes_data = json.loads(cpes_raw)
    assert len(cpes_data) >= 1
    assert cpes_data[0]["vendor"] == "example"
    assert cpes_data[0]["product"] == "product"

    # Second record has no CPEs
    cpes_raw_2 = dataset.get_column("cpes")[1]
    cpes_data_2 = json.loads(cpes_raw_2)
    assert cpes_data_2 == []

    # Third record has vendor:widget_library
    cpes_raw_3 = dataset.get_column("cpes")[2]
    cpes_data_3 = json.loads(cpes_raw_3)
    assert cpes_data_3[0]["vendor"] == "vendor"
    assert cpes_data_3[0]["product"] == "widget_library"


def test_nvd_extractor_parses_references():
    """NVDExtractor parses references into JSON-encoded column."""
    extractor = NVDExtractor("tests/fixtures/nvd_sample.json")
    dataset = extractor.extract()

    # References are JSON-encoded
    references = dataset.get_column("references")

    # First record has 2 references
    refs_0 = json.loads(references[0])
    assert len(refs_0) == 2

    # First reference is Vendor Advisory
    assert "Vendor Advisory" in refs_0[0]["tags"]
    assert refs_0[0]["url"] == "https://example.com/security/advisory-2024-001"

    # Second reference is Patch
    assert "Patch" in refs_0[1]["tags"]

    # Second record has 1 reference with no tags
    refs_1 = json.loads(references[1])
    assert len(refs_1) == 1
    assert refs_1[0]["tags"] == []


def test_nvd_extractor_handles_utf8():
    """NVDExtractor handles UTF-8 encoded files."""
    import tempfile
    from pathlib import Path

    # Create test file with UTF-8 characters
    nvd_data = {
        "resultsPerPage": 1,
        "startIndex": 0,
        "totalResults": 1,
        "format": "NVD_CVE",
        "version": "2.0",
        "vulnerabilities": [
            {
                "cve": {
                    "id": "CVE-2024-0001",
                    "descriptions": [
                        {"lang": "en", "value": "Test with UTF-8: café résumé"}
                    ],
                    "published": "2024-01-01T00:00:00.000",
                    "lastModified": "2024-01-01T00:00:00.000",
                }
            }
        ],
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(nvd_data, f)
        temp_path = f.name

    try:
        extractor = NVDExtractor(temp_path)
        dataset = extractor.extract()

        assert len(dataset) == 1
        descriptions = dataset.get_column("description")
        assert "café" in descriptions[0]
    finally:
        Path(temp_path).unlink()


def test_all_extractors_have_format_attribute():
    """All extractors have format attribute with correct values."""
    from vulnflow.extractors.files import ParquetExtractor

    # Test NVDExtractor
    assert hasattr(NVDExtractor, "format")
    assert NVDExtractor.format == "nvd_json"

    # Test ParquetExtractor
    assert hasattr(ParquetExtractor, "format")
    assert ParquetExtractor.format == "parquet"


# JSONExtractor tests


def test_json_extractor_roundtrip():
    """JSONExtractor roundtrip through JSONLoader preserves data."""
    import tempfile

    from vulnflow.loaders.files import JSONLoader

    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    with tempfile.TemporaryDirectory() as tmpdir:
        from pathlib import Path

        output_path = Path(tmpdir) / "output.json"
        loader = JSONLoader(output_path)

        # Write
        loader.load(dataset)

        # Read back
        restored = VulnDataset.from_json(output_path)

        # Verify
        assert len(restored) == len(dataset)
        assert set(restored.columns) == set(dataset.columns)

        # Check CVE IDs preserved
        original_cves = dataset.get_column("cve_id")
        restored_cves = restored.get_column("cve_id")
        assert list(original_cves) == list(restored_cves)


def test_json_extractor_schema_aware_datetime():
    """JSONExtractor reconstructs datetime types using Registry schema."""
    import tempfile
    from datetime import datetime
    from pathlib import Path

    import numpy as np

    from vulnflow.loaders.files import JSONLoader
    from vulnflow.registry import FieldDefinition, Registry

    # Register a datetime field
    Registry.register_schema(
        "TestComponent",
        [FieldDefinition("test_timestamp", "datetime", "Test datetime field")],
    )

    # Create dataset with datetime
    dataset = VulnDataset()
    test_time = datetime(2024, 1, 15, 12, 30, 45)
    dataset._data = {
        "cve_id": np.array(["CVE-2024-0001"], dtype=object),
        "test_timestamp": np.array([test_time], dtype=object),
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test.json"
        loader = JSONLoader(output_path)
        loader.load(dataset)

        # Read back
        restored = VulnDataset.from_json(output_path)

        # Verify datetime reconstructed
        restored_time = restored.get_column("test_timestamp")[0]
        assert isinstance(restored_time, datetime)
        assert restored_time == test_time


def test_json_extractor_schema_aware_uuid():
    """JSONExtractor reconstructs UUID types using Registry schema."""
    import tempfile
    from pathlib import Path
    from uuid import UUID, uuid4

    import numpy as np

    from vulnflow.loaders.files import JSONLoader
    from vulnflow.registry import FieldDefinition, Registry

    # Register a UUID field
    Registry.register_schema(
        "TestComponent2", [FieldDefinition("id", "uuid", "Test UUID field")]
    )

    # Create dataset with UUID
    dataset = VulnDataset()
    test_uuid = uuid4()
    dataset._data = {
        "cve_id": np.array(["CVE-2024-0001"], dtype=object),
        "id": np.array([test_uuid], dtype=object),
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test.json"
        loader = JSONLoader(output_path)
        loader.load(dataset)

        # Read back
        restored = VulnDataset.from_json(output_path)

        # Verify UUID reconstructed
        restored_uuid = restored.get_column("id")[0]
        assert isinstance(restored_uuid, UUID)
        assert restored_uuid == test_uuid


def test_json_extractor_fallback_inference():
    """JSONExtractor infers types for unregistered fields."""
    import tempfile
    from pathlib import Path

    # Create JSON with various unregistered types
    json_data = [
        {
            "cve_id": "CVE-2024-0001",
            "score": "9.8",  # Should become float
            "count": "42",  # Should become int
            "is_active": "True",  # Should become bool
            "description": "Some text",  # Should stay str
        }
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test.json"
        with open(output_path, "w") as f:
            json.dump(json_data, f)

        # Read
        dataset = VulnDataset.from_json(output_path)

        # Verify type inference
        assert isinstance(dataset.get_column("score")[0], float)
        assert dataset.get_column("score")[0] == 9.8

        assert isinstance(dataset.get_column("count")[0], int)
        assert dataset.get_column("count")[0] == 42

        assert isinstance(dataset.get_column("is_active")[0], bool)
        assert dataset.get_column("is_active")[0] is True

        assert isinstance(dataset.get_column("description")[0], str)


def test_json_extractor_handles_empty_file():
    """JSONExtractor handles empty JSON array."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "empty.json"
        with open(output_path, "w") as f:
            json.dump([], f)

        dataset = VulnDataset.from_json(output_path)
        assert len(dataset) == 0


def test_json_extractor_validates_json_format():
    """JSONExtractor validates JSON is a list."""
    import tempfile
    from pathlib import Path

    import pytest

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "invalid.json"
        with open(output_path, "w") as f:
            json.dump({"not": "a list"}, f)

        with pytest.raises(ExtractorError, match="Expected JSON array"):
            VulnDataset.from_json(output_path)


def test_json_extractor_preserves_column_order():
    """JSONExtractor preserves column order from first record."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "ordered.json"

        # Create JSON with specific column order in first record
        data = [
            {"id": "1", "cve_id": "CVE-1", "score": 7.5, "tags": "test"},
            {"severity": "HIGH", "cve_id": "CVE-2", "id": "2"},  # Different order
        ]

        with open(output_path, "w") as f:
            json.dump(data, f)

        dataset = VulnDataset.from_json(output_path)

        # Should preserve first record's order: id, cve_id, score, tags
        # Then append new columns: severity
        expected_order = ["id", "cve_id", "score", "tags", "severity"]
        assert dataset.columns == expected_order


def test_json_extractor_repeated_loads_stable():
    """JSONExtractor produces identical column order on repeated loads."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "stable.json"

        data = [
            {"a": 1, "b": 2, "c": 3},
            {"d": 4, "a": 1, "b": 2},
        ]

        with open(output_path, "w") as f:
            json.dump(data, f)

        # Load multiple times
        ds1 = VulnDataset.from_json(output_path)
        ds2 = VulnDataset.from_json(output_path)
        ds3 = VulnDataset.from_json(output_path)

        # All loads should have identical column order
        assert ds1.columns == ds2.columns
        assert ds2.columns == ds3.columns
        assert ds1.columns == ["a", "b", "c", "d"]


# PandasExtractor tests


def test_pandas_extractor_roundtrip():
    """PandasExtractor roundtrip through to_pandas preserves data."""
    from vulnflow.loaders.dataframe import to_pandas

    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    # Convert to pandas
    df = to_pandas(dataset)

    # Read back
    restored = VulnDataset.from_pandas(df)

    # Verify
    assert len(restored) == len(dataset)
    assert set(restored.columns) == set(dataset.columns)


def test_pandas_extractor_handles_empty_dataframe():
    """PandasExtractor handles empty DataFrame."""
    import pandas as pd

    df = pd.DataFrame()
    dataset = VulnDataset.from_pandas(df)

    assert len(dataset) == 0
    assert dataset.columns == []


def test_pandas_extractor_warns_missing_core_fields(caplog):
    """PandasExtractor warns if core fields missing."""
    import pandas as pd

    df = pd.DataFrame({"some_column": [1, 2, 3]})

    dataset = VulnDataset.from_pandas(df)

    assert "missing 'cve_id' column" in caplog.text
    assert "missing 'description' column" in caplog.text
    assert len(dataset) == 3


# PolarsExtractor tests


def test_polars_extractor_roundtrip():
    """PolarsExtractor roundtrip through to_polars preserves data."""
    from vulnflow.loaders.dataframe import to_polars

    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    # Convert to polars
    df = to_polars(dataset)

    # Read back
    restored = VulnDataset.from_polars(df)

    # Verify
    assert len(restored) == len(dataset)
    assert set(restored.columns) == set(dataset.columns)


def test_polars_extractor_delegates_to_pandas():
    """PolarsExtractor delegates to PandasExtractor."""
    import polars as pl

    from vulnflow.extractors.files import PolarsExtractor

    # Create test Polars DataFrame
    df = pl.DataFrame({"cve_id": ["CVE-2024-0001"], "description": ["Test"]})

    extractor = PolarsExtractor(df)

    # Should delegate to PandasExtractor (verify by checking it works)
    dataset = extractor.extract()

    assert len(dataset) == 1
    assert dataset.get_column("cve_id")[0] == "CVE-2024-0001"


# Task 5 tests: New CVSS fields and nvd_status


def test_nvd_extractor_sets_is_authoritative():
    """NVD extractor sets is_authoritative based on type=Primary."""
    extractor = NVDExtractor("tests/fixtures/nvd_sample.json")
    dataset = extractor.extract()

    cvss_raw = dataset.get_column("cvss_scores")[0]
    cvss_scores = json.loads(cvss_raw)

    # First record has Primary type, so should have is_authoritative=True
    auth_scores = [s for s in cvss_scores if s.get("is_authoritative")]
    assert len(auth_scores) >= 1
    assert auth_scores[0]["is_authoritative"] is True


def test_nvd_extractor_extracts_exploitability_and_impact_scores():
    """NVD extractor extracts exploitability_score and impact_score from CVSS."""
    extractor = NVDExtractor("tests/fixtures/nvd_sample.json")
    dataset = extractor.extract()

    cvss_raw = dataset.get_column("cvss_scores")[0]
    cvss_scores = json.loads(cvss_raw)

    # First record should have exploitabilityScore=3.9 and impactScore=5.9
    assert len(cvss_scores) >= 1
    assert cvss_scores[0].get("exploitability_score") == 3.9
    assert cvss_scores[0].get("impact_score") == 5.9


def test_nvd_extractor_extracts_nvd_status():
    """NVD extractor adds nvd_status column."""
    extractor = NVDExtractor("tests/fixtures/nvd_sample.json")
    dataset = extractor.extract()

    assert "nvd_status" in dataset.columns
    status = dataset.get_column("nvd_status")

    # First record has vulnStatus="Analyzed"
    assert status[0] == "analyzed"
    # Second record has vulnStatus="Awaiting Analysis"
    assert status[1] == "awaiting"
    # Third record has vulnStatus="Modified"
    assert status[2] == "modified"


def test_nvd_extractor_status_missing_defaults_to_analyzed():
    """NVD extractor infers 'analyzed' when vulnStatus missing (pre-2017 CVEs)."""
    extractor = NVDExtractor("tests/fixtures/nvd_sample.json")
    dataset = extractor.extract()

    status = dataset.get_column("nvd_status")
    # Fourth record has no vulnStatus → should infer "analyzed"
    assert status[3] == "analyzed"


def test_nvd_extractor_status_missing_rejected_description():
    """NVD extractor infers 'rejected' when vulnStatus missing and description has ** REJECTED **."""
    extractor = NVDExtractor("tests/fixtures/nvd_sample.json")
    dataset = extractor.extract()

    status = dataset.get_column("nvd_status")
    # Fifth record has no vulnStatus + ** REJECTED ** in description → "rejected"
    assert status[4] == "rejected"


def test_nvd_extractor_status_never_none():
    """NVD extractor never returns None for nvd_status."""
    extractor = NVDExtractor("tests/fixtures/nvd_sample.json")
    dataset = extractor.extract()

    statuses = dataset.get_column("nvd_status")
    for i, s in enumerate(statuses):
        assert s is not None, f"nvd_status[{i}] is None"
        assert isinstance(s, str), f"nvd_status[{i}] is not str: {type(s)}"


def test_nvd_extractor_extracts_fix_version():
    """NVD extractor captures versionEndExcluding as fix_version."""
    extractor = NVDExtractor("tests/fixtures/nvd_sample.json")
    dataset = extractor.extract()

    # First record has versionEndExcluding="2.0.0"
    cpes_raw = dataset.get_column("cpes")[0]
    cpes = json.loads(cpes_raw)

    # Find CPE entries with fix_version
    fix_versions = [c.get("fix_version") for c in cpes if c.get("fix_version")]
    assert len(fix_versions) >= 1
    assert fix_versions[0] == "2.0.0"


def test_nvd_extractor_no_flattened_columns():
    """NVD extractor should not include flattened columns (moved to transforms)."""
    extractor = NVDExtractor("tests/fixtures/nvd_sample.json")
    dataset = extractor.extract()

    # These columns should NOT exist (moved to CVSSFlattenTransform)
    assert "cvss_base_score" not in dataset.columns
    assert "cvss_version" not in dataset.columns

    # These columns should NOT exist (moved to CWEFlattenTransform)
    assert "primary_cwe_id" not in dataset.columns
    assert "primary_cwe_name" not in dataset.columns

    # These columns should NOT exist (moved to CPEFlattenTransform)
    assert "primary_vendor" not in dataset.columns
    assert "primary_product" not in dataset.columns


def test_nvd_extractor_encodes_complex_fields():
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


# ParquetExtractor tests


def test_parquet_extractor_returns_dataset():
    """ParquetExtractor.extract() returns VulnDataset."""
    # First create a parquet file (include all statuses for lossless roundtrip)
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    with tempfile.TemporaryDirectory() as tmpdir:
        parquet_path = Path(tmpdir) / "test.parquet"
        dataset.to_parquet(str(parquet_path))

        # Extract it
        extractor = ParquetExtractor(str(parquet_path))
        result = extractor.extract()

        assert isinstance(result, VulnDataset)
        assert len(result) == 5


def test_parquet_extractor_direct_load():
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


def test_parquet_extractor_file_not_found():
    """ParquetExtractor raises ExtractorError for missing file."""
    extractor = ParquetExtractor("nonexistent.parquet")

    with pytest.raises(ExtractorError, match="not found"):
        extractor.extract()


# BaseExtractor format validation tests


def test_base_extractor_validates_format_attribute():
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


def test_base_extractor_accepts_valid_format():
    """BaseExtractor does not raise if format is set."""

    class ValidFormatExtractor(BaseExtractor):
        format = "json"
        name = "ValidFormatExtractor"

        def extract(self) -> VulnDataset:
            return VulnDataset()

    # Should not raise
    extractor = ValidFormatExtractor()
    assert extractor.format == "json"


def test_base_extractor_rejects_empty_format():
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


# Provenance label tests


class TestProvenanceLabel:
    """Tests for extractor provenance labels."""

    def test_nvd_extractor_label_with_file(self) -> None:
        ext = NVDExtractor("data/nvdcve-2.0-2024.json.gz")
        assert ext.provenance_label == "NVDExtractor(nvdcve-2.0-2024.json.gz)"

    def test_nvd_extractor_label_with_url(self) -> None:
        ext = NVDExtractor(
            "https://nvd.nist.gov/feeds/json/cve/2.0/nvdcve-2.0-2024.json.gz"
        )
        assert ext.provenance_label == "NVDExtractor(nvdcve-2.0-2024.json.gz)"

    def test_parquet_extractor_label(self) -> None:
        ext = ParquetExtractor("data/library.parquet")
        assert ext.provenance_label == "ParquetExtractor(library.parquet)"

    def test_json_extractor_label(self) -> None:
        from vulnflow.extractors.files import JSONExtractor

        ext = JSONExtractor("data/export.json")
        assert ext.provenance_label == "JSONExtractor(export.json)"

    def test_nvd_extract_records_provenance(self, nvd_sample_path: Path) -> None:
        ext = NVDExtractor(str(nvd_sample_path))
        ds = ext.extract()
        assert any("NVDExtractor(" in e for e in ds.provenance["extractors"])

    def test_parquet_round_trip_records_provenance(self, tmp_path: Path) -> None:
        import numpy as np

        ds = VulnDataset()
        ds._data = {"cve_id": np.array(["CVE-2024-0001"])}
        ds.to_parquet(tmp_path / "test.parquet")

        ds2 = ParquetExtractor(str(tmp_path / "test.parquet")).extract()
        assert "ParquetExtractor(test.parquet)" in ds2.provenance["extractors"]


def test_nvd_extractor_parses_cvss_v40(nvd_sample_dataset: VulnDataset) -> None:
    """NVDExtractor parses CVSS v4.0 metrics with component fields."""
    # Third record (CVE-2024-0003) has v4.0, v3.1, and v2.0
    cvss_raw = nvd_sample_dataset.get_column("cvss_scores")[2]
    cvss_data = json.loads(cvss_raw)

    assert len(cvss_data) == 3

    v40 = next(e for e in cvss_data if e["version"] == "4.0")
    assert v40["base_score"] == 5.1
    assert v40["base_severity"] == "MEDIUM"  # From fixture's baseSeverity field
    assert v40["source"] == "nvd"
    assert v40["is_authoritative"] is True

    # v4.0 component fields
    assert v40["attack_vector"] == "NETWORK"
    assert v40["attack_requirements"] == "NONE"
    assert v40["user_interaction"] == "PASSIVE"
    assert v40["vuln_confidentiality_impact"] == "LOW"
    assert v40["sub_confidentiality_impact"] == "LOW"


class TestNVDExtractorProvenance:
    def test_provenance_date_range_populated(self, nvd_sample_path: Path) -> None:
        """NVDExtractor sets date_min/date_max from nvd_published_at."""
        extractor = NVDExtractor(str(nvd_sample_path))
        ds = extractor.extract()
        prov = ds.provenance
        assert prov["date_min"] is not None
        assert prov["date_max"] is not None
        assert prov["date_min"] <= prov["date_max"]

    def test_provenance_date_range_none_for_empty(self, tmp_path: Path) -> None:
        """Empty dataset has None date_min/date_max."""
        empty = tmp_path / "empty.json"
        empty.write_text('{"vulnerabilities": []}')
        extractor = NVDExtractor(str(empty))
        ds = extractor.extract()
        prov = ds.provenance
        assert prov.get("date_min") is None
        assert prov.get("date_max") is None

    def test_parquet_roundtrip_preserves_date_range(
        self, tmp_path: Path, nvd_sample_path: Path
    ) -> None:
        """date_min/date_max survive parquet round-trip."""
        ds = NVDExtractor(str(nvd_sample_path)).extract()
        original_min = ds._provenance.get("date_min")
        original_max = ds._provenance.get("date_max")
        assert original_min is not None  # Prerequisite from Task 1

        ds.to_parquet(tmp_path / "test.parquet")
        ds2 = VulnDataset.from_parquet(str(tmp_path / "test.parquet"))

        assert ds2._provenance.get("date_min") == original_min
        assert ds2._provenance.get("date_max") == original_max


# Compression format tests


def test_nvd_extractor_reads_zip(nvd_sample_path: Path) -> None:
    """NVDExtractor reads .json.zip files correctly."""
    import zipfile

    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "nvd_sample.json.zip"
        raw = nvd_sample_path.read_bytes()
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("nvd_sample.json", raw)

        extractor = NVDExtractor(str(zip_path))
        dataset = extractor.extract()

        assert len(dataset) == 5
        assert "cve_id" in dataset.columns


def test_nvd_extractor_zip_url_decompresses(nvd_sample_path: Path) -> None:
    """NVDExtractor decompresses zip content fetched from URL-like .zip path."""
    import zipfile
    from io import BytesIO

    # Create zip bytes in memory
    raw = nvd_sample_path.read_bytes()
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("nvdcve-2.0-modified.json", raw)
    zip_bytes = buf.getvalue()

    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "nvdcve-2.0-modified.json.zip"
        zip_path.write_bytes(zip_bytes)

        extractor = NVDExtractor(str(zip_path))
        dataset = extractor.extract()

        assert len(dataset) == 5
        assert "cve_id" in dataset.columns


def test_nvd_extractor_reads_gzip(nvd_sample_path: Path) -> None:
    """NVDExtractor reads .json.gz files correctly."""
    import gzip

    with tempfile.TemporaryDirectory() as tmpdir:
        gz_path = Path(tmpdir) / "nvd_sample.json.gz"
        raw = nvd_sample_path.read_bytes()
        gz_path.write_bytes(gzip.compress(raw))

        extractor = NVDExtractor(str(gz_path))
        dataset = extractor.extract()

        assert len(dataset) == 5
        assert "cve_id" in dataset.columns


def test_nvd_extractor_reads_jsonl(nvd_sample_path: Path) -> None:
    """NVDExtractor reads .jsonl (newline-delimited JSON) files."""
    raw = nvd_sample_path.read_bytes()
    data = json.loads(raw)
    vulns = data["vulnerabilities"]

    with tempfile.TemporaryDirectory() as tmpdir:
        jsonl_path = Path(tmpdir) / "nvd_sample.jsonl"
        lines = [json.dumps(v) for v in vulns]
        jsonl_path.write_text("\n".join(lines), encoding="utf-8")

        extractor = NVDExtractor(str(jsonl_path))
        dataset = extractor.extract()

        assert len(dataset) == 5
        assert "cve_id" in dataset.columns


def test_nvd_extractor_corrupt_gzip_raises() -> None:
    """NVDExtractor raises ExtractorError on corrupt .gz file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        corrupt_path = Path(tmpdir) / "corrupt.json.gz"
        # Write gzip magic bytes followed by garbage
        corrupt_path.write_bytes(b"\x1f\x8b" + b"\x00" * 20)

        extractor = NVDExtractor(str(corrupt_path))
        with pytest.raises(ExtractorError, match="Failed to decompress"):
            extractor.extract()


# URL download path tests


def test_nvd_extractor_url_path(nvd_sample_path: Path) -> None:
    """NVDExtractor uses HttpFetcher for URL paths."""
    from unittest.mock import patch

    raw = nvd_sample_path.read_bytes()

    with patch("vulnflow.extractors.nvd.is_url", return_value=True):
        with patch("vulnflow.extractors.nvd.HttpFetcher") as MockFetcher:
            instance = MockFetcher.return_value
            instance.fetch_sync.return_value = (raw, "json")

            extractor = NVDExtractor("https://example.com/nvd.json")
            dataset = extractor.extract()

            instance.fetch_sync.assert_called_once()
            assert len(dataset) == 5


# Parquet multi-file upsert tests


def test_parquet_extractor_multi_file_upsert(tmp_path: Path) -> None:
    """ParquetExtractor deduplicates overlapping CVE IDs across files (later wins)."""
    import numpy as np

    ds1 = VulnDataset()
    ds1._data = {
        "cve_id": np.array(["CVE-2024-0001", "CVE-2024-0002"], dtype=object),
        "description": np.array(["old-1", "old-2"], dtype=object),
    }
    ds1.to_parquet(tmp_path / "file1.parquet")

    ds2 = VulnDataset()
    ds2._data = {
        "cve_id": np.array(["CVE-2024-0002", "CVE-2024-0003"], dtype=object),
        "description": np.array(["new-2", "new-3"], dtype=object),
    }
    ds2.to_parquet(tmp_path / "file2.parquet")

    extractor = ParquetExtractor([
        str(tmp_path / "file1.parquet"),
        str(tmp_path / "file2.parquet"),
    ])
    merged = extractor.extract()

    assert len(merged) == 3
    cve_ids = list(merged.get_column("cve_id"))
    descriptions = list(merged.get_column("description"))

    # CVE-2024-0002 should have the value from file2 (later wins)
    idx = cve_ids.index("CVE-2024-0002")
    assert descriptions[idx] == "new-2"


# Malformed input tests


@pytest.mark.parametrize(
    "content,suffix,error_type",
    [
        pytest.param(
            b"not valid json {{{",
            ".json",
            ExtractorError,
            id="invalid_json",
        ),
        pytest.param(
            b"",
            ".json",
            ExtractorError,
            id="empty_file",
        ),
        pytest.param(
            json.dumps({"format": "NVD_CVE", "version": "2.0"}).encode(),
            ".json",
            None,  # No error — returns empty dataset with warning
            id="json_no_vulnerabilities_key",
        ),
    ],
)
def test_nvd_extractor_malformed_input(
    content: bytes, suffix: str, error_type: type | None
) -> None:
    """NVDExtractor handles malformed inputs gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / f"malformed{suffix}"
        path.write_bytes(content)

        extractor = NVDExtractor(str(path))
        if error_type is not None:
            with pytest.raises(error_type):
                extractor.extract()
        else:
            dataset = extractor.extract()
            assert len(dataset) == 0


# HTTP 404 and unrecognized content tests


def test_nvd_extractor_missing_file_raises_extractor_error() -> None:
    """NVDExtractor raises ExtractorError for missing local file."""
    extractor = NVDExtractor("nonexistent_file_12345.json")
    with pytest.raises(ExtractorError, match="nonexistent_file_12345"):
        extractor.extract()


def test_nvd_extractor_http_404_raises_extractor_error() -> None:
    """NVDExtractor wraps HTTP 404 into clean ExtractorError without chained tracebacks."""
    from unittest.mock import patch

    from vulnflow.utils.http import HttpError

    with patch("vulnflow.extractors.nvd.is_url", return_value=True):
        with patch("vulnflow.extractors.nvd.HttpFetcher") as MockFetcher:
            instance = MockFetcher.return_value
            instance.fetch_sync.side_effect = HttpError(
                "Failed to fetch https://example.com/nvd.json: 404 Not Found"
            )

            extractor = NVDExtractor("https://example.com/nvd.json")
            with pytest.raises(ExtractorError, match="404") as exc_info:
                extractor.extract()
            # Chain must be suppressed (from None) so Jupyter shows one traceback
            assert exc_info.value.__cause__ is None


def test_nvd_extractor_http_error_wraps_as_extractor_error() -> None:
    """NVDExtractor wraps generic HTTP errors into ExtractorError."""
    from unittest.mock import patch

    from vulnflow.utils.http import HttpError

    with patch("vulnflow.extractors.nvd.is_url", return_value=True):
        with patch("vulnflow.extractors.nvd.HttpFetcher") as MockFetcher:
            instance = MockFetcher.return_value
            instance.fetch_sync.side_effect = HttpError("Connection refused")

            extractor = NVDExtractor("https://example.com/nvd.json")
            with pytest.raises(ExtractorError, match="Connection refused"):
                extractor.extract()


def test_nvd_extractor_unrecognized_binary_raises_extractor_error() -> None:
    """NVDExtractor raises ExtractorError for unrecognized binary content."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "mystery.json"
        # Write random binary that isn't gz, zip, or valid JSON
        path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        extractor = NVDExtractor(str(path))
        with pytest.raises(ExtractorError, match="Failed to parse"):
            extractor.extract()


def test_nvd_extractor_corrupt_zip_raises_extractor_error() -> None:
    """NVDExtractor raises ExtractorError on corrupt .zip file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        corrupt_path = Path(tmpdir) / "corrupt.json.zip"
        # Write zip magic bytes followed by garbage
        corrupt_path.write_bytes(b"PK\x03\x04" + b"\x00" * 20)

        extractor = NVDExtractor(str(corrupt_path))
        with pytest.raises(ExtractorError, match="Failed to decompress zip"):
            extractor.extract()
