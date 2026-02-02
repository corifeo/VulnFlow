"""
Schema Registration Tests
Tests: SCHEMA_VERSION and OUTPUT_SCHEMA presence on transforms

Layer: Transform (ETL)
Dependencies: transforms
"""

from __future__ import annotations


def test_kev_enricher_has_schema_version():
    """KEVEnricher defines SCHEMA_VERSION."""
    from vulnflow.transforms.enrichers.kev import KEVEnricher

    assert hasattr(KEVEnricher, "SCHEMA_VERSION")
    assert isinstance(KEVEnricher.SCHEMA_VERSION, str)
    assert KEVEnricher.SCHEMA_VERSION  # Not empty


def test_kev_enricher_has_output_schema():
    """KEVEnricher defines OUTPUT_SCHEMA."""
    from vulnflow.transforms.enrichers.kev import KEVEnricher

    assert hasattr(KEVEnricher, "OUTPUT_SCHEMA")
    assert isinstance(KEVEnricher.OUTPUT_SCHEMA, list)
    assert len(KEVEnricher.OUTPUT_SCHEMA) > 0

    # Verify schema structure
    for field in KEVEnricher.OUTPUT_SCHEMA:
        assert "name" in field
        assert "type" in field


def test_epss_enricher_has_schema_version():
    """EPSSEnricher defines SCHEMA_VERSION."""
    from vulnflow.transforms.enrichers.epss import EPSSEnricher

    assert hasattr(EPSSEnricher, "SCHEMA_VERSION")
    assert isinstance(EPSSEnricher.SCHEMA_VERSION, str)


def test_epss_enricher_has_output_schema():
    """EPSSEnricher defines OUTPUT_SCHEMA."""
    from vulnflow.transforms.enrichers.epss import EPSSEnricher

    assert hasattr(EPSSEnricher, "OUTPUT_SCHEMA")
    assert isinstance(EPSSEnricher.OUTPUT_SCHEMA, list)
    assert len(EPSSEnricher.OUTPUT_SCHEMA) >= 4  # probability, percentile, date, model


def test_event_inference_has_schema():
    """EventInference defines SCHEMA_VERSION and OUTPUT_SCHEMA."""
    from vulnflow.transforms.inferences.events import EventInference

    assert hasattr(EventInference, "SCHEMA_VERSION")
    assert hasattr(EventInference, "OUTPUT_SCHEMA")
    assert len(EventInference.OUTPUT_SCHEMA) >= 6  # 6 CVD timestamp columns


def test_tag_inference_has_schema():
    """TagInference defines SCHEMA_VERSION and OUTPUT_SCHEMA."""
    from vulnflow.transforms.inferences.tags import TagInference

    assert hasattr(TagInference, "SCHEMA_VERSION")
    assert hasattr(TagInference, "OUTPUT_SCHEMA")


def test_cvd_state_transform_has_schema():
    """CVDStateTransform defines SCHEMA_VERSION and OUTPUT_SCHEMA."""
    from vulnflow.transforms.computed.cvd_state import CVDStateTransform

    assert hasattr(CVDStateTransform, "SCHEMA_VERSION")
    assert hasattr(CVDStateTransform, "OUTPUT_SCHEMA")


def test_desiderata_transform_has_schema():
    """CVDDesiderataTransform defines SCHEMA_VERSION and OUTPUT_SCHEMA."""
    from vulnflow.transforms.computed.desiderata import CVDDesiderataTransform

    assert hasattr(CVDDesiderataTransform, "SCHEMA_VERSION")
    assert hasattr(CVDDesiderataTransform, "OUTPUT_SCHEMA")
