"""
Schema Registry - Track component output schemas
Tests: register_schema(), get_registered_fields()

Layer: Core
Dependencies: None
"""

from __future__ import annotations

import pytest
from vulnflow.schema_registry import FieldDefinition, SchemaRegistry


def test_register_schema():
    """register_schema() registers component output fields."""
    registry = SchemaRegistry()

    fields = [
        FieldDefinition(name="kev_date_added", type="datetime", source="KEVEnricher"),
        FieldDefinition(name="kev_vendor", type="str", source="KEVEnricher"),
    ]

    registry.register_schema("KEVEnricher", "1.0.0", fields)

    registered = registry.get_registered_fields()
    assert "kev_date_added" in registered
    assert "kev_vendor" in registered


def test_get_registered_fields_returns_dict():
    """get_registered_fields() returns field name -> FieldDefinition mapping."""
    registry = SchemaRegistry()

    fields = [
        FieldDefinition(name="epss_probability", type="float", source="EPSSEnricher"),
    ]

    registry.register_schema("EPSSEnricher", "1.0.0", fields)

    registered = registry.get_registered_fields()
    assert isinstance(registered, dict)
    assert registered["epss_probability"].type == "float"
    assert registered["epss_probability"].source == "EPSSEnricher"


def test_get_field_providers():
    """get_field_providers() returns which component provides each field."""
    registry = SchemaRegistry()

    registry.register_schema(
        "KEVEnricher",
        "1.0.0",
        [
            FieldDefinition(name="kev_date_added", type="datetime", source="KEVEnricher"),
        ],
    )

    registry.register_schema(
        "EPSSEnricher",
        "1.0.0",
        [
            FieldDefinition(name="epss_probability", type="float", source="EPSSEnricher"),
        ],
    )

    providers = registry.get_field_providers()
    assert providers["kev_date_added"] == "KEVEnricher"
    assert providers["epss_probability"] == "EPSSEnricher"


def test_clear_registry():
    """clear() removes all registrations."""
    registry = SchemaRegistry()

    registry.register_schema(
        "Test",
        "1.0.0",
        [
            FieldDefinition(name="test_field", type="str", source="Test"),
        ],
    )

    assert len(registry.get_registered_fields()) == 1

    registry.clear()
    assert len(registry.get_registered_fields()) == 0
