"""Tests that verify transform hierarchy is enforced."""
import pytest
from vulnflow.transforms.enrichers.kev import KEVEnricher
from vulnflow.transforms.enrichers.epss import EPSSEnricher
from vulnflow.transforms.base import BaseEnricher, BaseTransform
from vulnflow.registry import FieldDefinition, Registry


def test_all_enrichers_inherit_from_base_enricher():
    """All enrichers must inherit from BaseEnricher."""
    enrichers = [KEVEnricher, EPSSEnricher]

    for enricher_cls in enrichers:
        assert issubclass(enricher_cls, BaseEnricher), \
            f"{enricher_cls.__name__} must inherit from BaseEnricher"


def test_all_enrichers_inherit_from_base_transform():
    """All enrichers inherit from BaseTransform (via BaseEnricher)."""
    enrichers = [KEVEnricher, EPSSEnricher]

    for enricher_cls in enrichers:
        assert issubclass(enricher_cls, BaseTransform), \
            f"{enricher_cls.__name__} must inherit from BaseTransform"


def test_all_enrichers_use_field_definition():
    """All enrichers must use FieldDefinition in OUTPUT_SCHEMA."""
    enrichers = [KEVEnricher, EPSSEnricher]

    for enricher_cls in enrichers:
        schema = enricher_cls.OUTPUT_SCHEMA
        assert isinstance(schema, list), \
            f"{enricher_cls.__name__}.OUTPUT_SCHEMA must be a list"

        for field in schema:
            assert isinstance(field, FieldDefinition), \
                f"{enricher_cls.__name__}.OUTPUT_SCHEMA must use FieldDefinition, got {type(field)}"


def test_all_enrichers_register_schema():
    """All enrichers register schema on init."""
    Registry.clear()

    kev = KEVEnricher(source="tests/fixtures/kev_sample.csv")
    epss = EPSSEnricher(source="tests/fixtures/epss_sample.csv")

    kev_cols = Registry.get_component_columns("KEVEnricher")
    assert len(kev_cols) == 6

    epss_cols = Registry.get_component_columns("EPSSEnricher")
    assert len(epss_cols) == 4

    all_cols = Registry.get_all_columns()
    assert len(all_cols) == 10


def test_enricher_schema_has_source_component():
    """All FieldDefinitions specify source_component."""
    Registry.clear()

    kev = KEVEnricher(source="tests/fixtures/kev_sample.csv")

    cols = Registry.get_component_columns("KEVEnricher")
    for col in cols:
        assert col.source_component == "KEVEnricher"


def test_enricher_has_async_transform():
    """All enrichers have async transform method."""
    import inspect

    enrichers = [KEVEnricher, EPSSEnricher]

    for enricher_cls in enrichers:
        assert hasattr(enricher_cls, 'transform')
        assert inspect.iscoroutinefunction(enricher_cls.transform), \
            f"{enricher_cls.__name__}.transform must be async"
