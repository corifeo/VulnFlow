"""
Tag Inference - Infer vulnerability classification tags
Tests: Threat type, attack vector, product categorization

Layer: Transform/Inferences
Dependencies: base, schema
"""

from __future__ import annotations

from tests.conftest import dataset_from_records

from vulnflow.schema import CPEEntry, CWEEntry, VulnRecord
from vulnflow.transforms.inferences.tags import TagInference


def test_tag_inference_threat_from_description():
    """TagInference infers threat tags from keywords."""
    records = [
        VulnRecord(
            cve_id="CVE-2024-0001",
            description="A SQL injection vulnerability allows remote code execution.",
        )
    ]
    dataset = dataset_from_records(records)

    TagInference(confidence_threshold=0.7).transform(dataset)

    # Check tags column exists and has threat tags
    assert "inferred_tags" in dataset.columns
    tags = dataset.get_column("inferred_tags")[0]
    assert "sqli" in tags
    assert "rce" in tags


def test_tag_inference_vector_from_cwe():
    """TagInference infers attack vector from CWE."""
    records = [
        VulnRecord(
            cve_id="CVE-2024-0001",
            description="Test",
            cwes=[CWEEntry(id="CWE-89", name="SQL Injection", category_id="CWE-1000")],
        )
    ]
    dataset = dataset_from_records(records)

    TagInference().transform(dataset)

    tags = dataset.get_column("inferred_tags")[0]
    assert "web" in tags


def test_tag_inference_product_from_cpe():
    """TagInference infers product type from CPE."""
    records = [
        VulnRecord(
            cve_id="CVE-2024-0001",
            description="Test",
            cpes=[
                CPEEntry(
                    vendor="apache",
                    product="http_server",
                    version="2.4",
                    raw="cpe:2.3:a:apache:http_server:2.4:*:*:*:*:*:*:*",
                )
            ],
        )
    ]
    dataset = dataset_from_records(records)

    TagInference().transform(dataset)

    tags = dataset.get_column("inferred_tags")[0]
    assert "web-server" in tags


def test_tag_inference_confidence_threshold():
    """TagInference respects confidence threshold."""
    records = [
        VulnRecord(
            cve_id="CVE-2024-0001",
            description="SQL injection",
        )
    ]
    dataset = dataset_from_records(records)

    # Threshold > 1.0 should skip inference
    TagInference(confidence_threshold=1.5).transform(dataset)

    # Column should exist but be empty
    tags = dataset.get_column("inferred_tags")[0]
    assert len(tags) == 0


def test_tag_inference_has_required_attributes():
    """TagInference has required transform attributes."""
    assert TagInference.name == "TagInference"
    assert hasattr(TagInference, "depends_on")

    # confidence_threshold is set in __init__
    instance = TagInference()
    assert instance.confidence_threshold == 0.7
