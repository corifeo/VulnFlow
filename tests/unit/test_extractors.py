"""
Extractors - NVDExtractor and base protocol
Tests: NVDExtractor parsing, Extractor protocol

Layer: Extract (ETL)
Dependencies: schema
"""

from __future__ import annotations

import json

from vulnflow.dataset import VulnDataset
from vulnflow.extractors.nvd import NVDExtractor


def test_nvd_extractor_returns_records():
    """NVDExtractor.extract() returns VulnDataset."""
    extractor = NVDExtractor("tests/fixtures/nvd_sample.json")
    dataset = extractor.extract()

    assert isinstance(dataset, VulnDataset)
    assert len(dataset) == 3
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
    """NVDExtractor parses CVSS v3 scores and flattens them."""
    extractor = NVDExtractor("tests/fixtures/nvd_sample.json")
    dataset = extractor.extract()

    # First record has CVSS (flattened)
    cvss_scores = dataset.get_column("cvss_base_score")
    assert cvss_scores[0] == 9.8

    cvss_severity = dataset.get_column("cvss_base_severity")
    assert cvss_severity[0] == "CRITICAL"

    cvss_source = dataset.get_column("cvss_source")
    assert cvss_source[0] == "nvd"

    # Second record (CVE-2024-0002) has no CVSS metrics
    assert cvss_scores[1] is None


def test_nvd_extractor_parses_cwes():
    """NVDExtractor parses CWE weaknesses and flattens them."""
    extractor = NVDExtractor("tests/fixtures/nvd_sample.json")
    dataset = extractor.extract()

    # First record has CWE-94
    primary_cwe = dataset.get_column("primary_cwe_id")
    assert primary_cwe[0] == "CWE-94"

    # Second record has no CWEs
    assert primary_cwe[1] is None

    # Third record has CWE-79
    assert primary_cwe[2] == "CWE-79"


def test_nvd_extractor_parses_cpes():
    """NVDExtractor parses CPE configurations and flattens them."""
    extractor = NVDExtractor("tests/fixtures/nvd_sample.json")
    dataset = extractor.extract()

    # First record has CPE for example:product
    primary_vendor = dataset.get_column("primary_vendor")
    primary_product = dataset.get_column("primary_product")

    assert primary_vendor[0] == "example"
    assert primary_product[0] == "product"

    # Second record has no CPEs
    assert primary_vendor[1] is None
    assert primary_product[1] is None

    # Third record has vendor:widget_library
    assert primary_vendor[2] == "vendor"
    assert primary_product[2] == "widget_library"


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

        with pytest.raises(ValueError, match="Expected JSON array"):
            VulnDataset.from_json(output_path)


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
