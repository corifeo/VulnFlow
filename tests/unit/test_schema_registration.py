"""
Schema Registration Tests
Tests: SCHEMA_VERSION and OUTPUT_SCHEMA presence on transforms

Layer: Transform (ETL)
Dependencies: transforms
"""

from __future__ import annotations


def test_kev_enricher_has_output_schema():
    """KEVEnricher defines OUTPUT_SCHEMA with FieldDefinitions."""
    from vulnflow.registry import FieldDefinition
    from vulnflow.transforms.enrichers.kev import KEVEnricher

    assert hasattr(KEVEnricher, "OUTPUT_SCHEMA")
    assert isinstance(KEVEnricher.OUTPUT_SCHEMA, list)
    assert len(KEVEnricher.OUTPUT_SCHEMA) > 0

    # Verify schema structure (new FieldDefinition format)
    for field in KEVEnricher.OUTPUT_SCHEMA:
        assert isinstance(field, FieldDefinition)
        assert field.name
        assert field.type


def test_epss_enricher_has_output_schema():
    """EPSSEnricher defines OUTPUT_SCHEMA with FieldDefinitions."""
    from vulnflow.registry import FieldDefinition
    from vulnflow.transforms.enrichers.epss import EPSSEnricher

    assert hasattr(EPSSEnricher, "OUTPUT_SCHEMA")
    assert isinstance(EPSSEnricher.OUTPUT_SCHEMA, list)
    assert len(EPSSEnricher.OUTPUT_SCHEMA) >= 4  # probability, percentile, date, model

    # Verify schema structure (new FieldDefinition format)
    for field in EPSSEnricher.OUTPUT_SCHEMA:
        assert isinstance(field, FieldDefinition)
        assert field.name
        assert field.type


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
    """CVDStateTransform defines SCHEMA_VERSION and OUTPUT_SCHEMA with all 19 fields."""
    from vulnflow.transforms.computed.cvd_state import CVDStateTransform

    assert hasattr(CVDStateTransform, "SCHEMA_VERSION")
    assert hasattr(CVDStateTransform, "OUTPUT_SCHEMA")
    # CVDStateTransform now computes all 19 CVD fields (including desiderata)
    assert len(CVDStateTransform.OUTPUT_SCHEMA) == 19


def test_cvd_state_transform_includes_desiderata_fields():
    """CVDStateTransform OUTPUT_SCHEMA includes desiderata_mask and desiderata_count."""
    from vulnflow.transforms.computed.cvd_state import CVDStateTransform

    field_names = [f.name for f in CVDStateTransform.OUTPUT_SCHEMA]

    # Core fields
    assert "cvd_state" in field_names
    assert "state_bits" in field_names

    # Desiderata fields (consolidated from old CVDDesiderataTransform)
    assert "desiderata_mask" in field_names
    assert "desiderata_count" in field_names

    # Zero-day flags
    assert "is_zero_day_exploit" in field_names
    assert "is_zero_day_attack" in field_names
