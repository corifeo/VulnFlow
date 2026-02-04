"""
Multi-Source CVSS Tests - Verify CVSS source parsing from NVD metrics

Layer: Unit (Extractor)
Dependencies: schema, extractors, transforms.flatten
"""

from __future__ import annotations

import json

from vulnflow.extractors.nvd import NVDExtractor


def test_nvd_extractor_parses_cvss_source():
    """NVDExtractor extracts source from CVSS metrics into nested array."""
    extractor = NVDExtractor("tests/fixtures/nvd_sample.json")
    dataset = extractor.extract()

    # First record should have CVSS scores in nested array
    assert len(dataset) > 0

    # Check that source is extracted in nested array
    cvss_scores_json = dataset.get_column("cvss_scores")[0]
    cvss_scores = json.loads(cvss_scores_json)
    assert len(cvss_scores) > 0
    assert cvss_scores[0]["source"] is not None
    # NVD sample has "nvd@nist.gov" which normalizes to "nvd"
    assert cvss_scores[0]["source"] == "nvd"


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
    """CVSSFlattenTransform produces cvss_source in flattened columns."""
    from vulnflow import VulnDataset
    from vulnflow.transforms.flatten import CVSSFlattenTransform

    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")
    CVSSFlattenTransform().transform(dataset)
    df = dataset.to_pandas()

    # Check flattened CVSS source column exists
    assert "cvss_source" in df.columns

    # First record should have nvd as source
    assert df["cvss_source"].iloc[0] == "nvd"
