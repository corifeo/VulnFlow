"""
Multi-Source CVSS Tests - Verify CVSS source parsing from NVD metrics

Layer: Unit (Extractor)
Dependencies: schema, extractors
"""

from __future__ import annotations

from vulnflow.extractors.nvd import NVDExtractor


def test_nvd_extractor_parses_cvss_source():
    """NVDExtractor extracts source from CVSS metrics."""
    extractor = NVDExtractor("tests/fixtures/nvd_sample.json")
    records = extractor.extract()

    # First record should have CVSS scores
    assert len(records) > 0
    assert len(records[0].cvss_scores) > 0

    # Check that source is extracted (not hardcoded as "nvd")
    cvss = records[0].cvss_scores[0]
    assert cvss.source is not None
    assert isinstance(cvss.source, str)
    # NVD sample has "nvd@nist.gov" which normalizes to "nvd"
    assert cvss.source == "nvd"


def test_normalize_cvss_source():
    """_normalize_cvss_source correctly maps email to identifier."""
    extractor = NVDExtractor("tests/fixtures/nvd_sample.json")

    # Test known sources
    assert extractor._normalize_cvss_source("nvd@nist.gov") == "nvd"
    assert extractor._normalize_cvss_source("security@redhat.com") == "redhat"
    assert extractor._normalize_cvss_source("security@microsoft.com") == "microsoft"
    assert extractor._normalize_cvss_source("cve@mitre.org") == "mitre"

    # Test domain extraction
    assert extractor._normalize_cvss_source("security@example.com") == "example"


def test_cvss_source_flattening():
    """DataFrame export uses CVSS source in flattened columns."""
    from vulnflow import VulnDataset

    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")
    df = dataset.to_pandas()

    # Check flattened CVSS source column exists
    assert "cvss_source" in df.columns

    # First record should have nvd as source
    assert df["cvss_source"].iloc[0] == "nvd"
