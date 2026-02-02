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
