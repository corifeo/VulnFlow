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
        "vulnerabilities": [{
            "cve": {
                "id": "CVE-2024-0001",
                "descriptions": [{
                    "lang": "en",
                    "value": "Test with UTF-8: café résumé"
                }],
                "published": "2024-01-01T00:00:00.000",
                "lastModified": "2024-01-01T00:00:00.000"
            }
        }]
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
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
    assert hasattr(NVDExtractor, 'format')
    assert NVDExtractor.format == "nvd_json"

    # Test ParquetExtractor
    assert hasattr(ParquetExtractor, 'format')
    assert ParquetExtractor.format == "parquet"
