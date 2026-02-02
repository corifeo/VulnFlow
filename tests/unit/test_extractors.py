"""
Extractors - NVDExtractor and base protocol
Tests: NVDExtractor parsing, Extractor protocol

Layer: Extract (ETL)
Dependencies: schema
"""

from __future__ import annotations

from vulnflow.extractors.nvd import NVDExtractor
from vulnflow.schema import VulnRecord


def test_nvd_extractor_returns_records():
    """NVDExtractor.extract() returns list of VulnRecord."""
    extractor = NVDExtractor("tests/fixtures/nvd_sample.json")
    records = extractor.extract()

    assert isinstance(records, list)
    assert len(records) == 3
    assert all(isinstance(r, VulnRecord) for r in records)


def test_nvd_extractor_parses_cve_id():
    """NVDExtractor correctly parses CVE IDs."""
    extractor = NVDExtractor("tests/fixtures/nvd_sample.json")
    records = extractor.extract()

    assert records[0].cve_id == "CVE-2024-0001"
    assert records[1].cve_id == "CVE-2024-0002"
    assert records[2].cve_id == "CVE-2024-0003"


def test_nvd_extractor_parses_nvd_timestamps():
    """NVDExtractor populates nvd_published_at and nvd_modified_at."""
    extractor = NVDExtractor("tests/fixtures/nvd_sample.json")
    records = extractor.extract()

    # First record has timestamps
    assert records[0].nvd_published_at is not None
    assert records[0].nvd_published_at.year == 2024
    assert records[0].nvd_modified_at is not None


def test_nvd_extractor_parses_cvss():
    """NVDExtractor parses CVSS v3 scores into cvss_scores list."""
    extractor = NVDExtractor("tests/fixtures/nvd_sample.json")
    records = extractor.extract()

    # First record has CVSS
    assert len(records[0].cvss_scores) > 0
    cvss = records[0].cvss_scores[0]
    assert cvss.base_score == 9.8
    assert cvss.base_severity == "CRITICAL"
    assert cvss.source == "nvd"

    # Second record (CVE-2024-0002) has no CVSS metrics
    assert len(records[1].cvss_scores) == 0


def test_nvd_extractor_parses_cwes():
    """NVDExtractor parses CWE weaknesses as CWEEntry list."""
    extractor = NVDExtractor("tests/fixtures/nvd_sample.json")
    records = extractor.extract()

    # First record has CWE-94
    assert len(records[0].cwes) > 0
    cwe = records[0].cwes[0]
    assert cwe.id == "CWE-94"

    # Second record has no CWEs
    assert len(records[1].cwes) == 0

    # Third record has CWE-79
    assert len(records[2].cwes) > 0
    assert records[2].cwes[0].id == "CWE-79"


def test_nvd_extractor_parses_cpes():
    """NVDExtractor parses CPE configurations as CPEEntry list."""
    extractor = NVDExtractor("tests/fixtures/nvd_sample.json")
    records = extractor.extract()

    # First record has CPE for example:product
    assert len(records[0].cpes) > 0
    cpe = records[0].cpes[0]
    assert cpe.vendor == "example"
    assert cpe.product == "product"
    assert cpe.raw is not None
    assert cpe.raw.startswith("cpe:2.3")

    # Second record has no CPEs
    assert len(records[1].cpes) == 0

    # Third record has 2 CPEs for widget_library
    assert len(records[2].cpes) == 2
    assert records[2].cpes[0].vendor == "vendor"
    assert records[2].cpes[0].product == "widget_library"


def test_nvd_extractor_parses_references():
    """NVDExtractor parses references with tags."""
    extractor = NVDExtractor("tests/fixtures/nvd_sample.json")
    records = extractor.extract()

    # First record has 2 references
    assert len(records[0].references) == 2

    # First reference is Vendor Advisory
    ref1 = records[0].references[0]
    assert "Vendor Advisory" in ref1.tags
    assert ref1.url == "https://example.com/security/advisory-2024-001"

    # Second reference is Patch
    ref2 = records[0].references[1]
    assert "Patch" in ref2.tags

    # Second record has 1 reference with no tags
    assert len(records[1].references) == 1
    assert records[1].references[0].tags == []
