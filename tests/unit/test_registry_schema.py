"""Tests for schema registry functionality."""

from vulnflow.registry import FieldDefinition, Registry


def test_field_definition_creation():
    """FieldDefinition dataclass can be created."""
    field = FieldDefinition(
        name="kev_date_added",
        type="datetime",
        description="Date added to KEV catalog",
        source_component="KEVEnricher",
    )

    assert field.name == "kev_date_added"
    assert field.type == "datetime"
    assert field.description == "Date added to KEV catalog"
    assert field.source_component == "KEVEnricher"


def test_register_schema():
    """Registry.register_schema() stores schema."""
    Registry.clear()

    fields = [
        FieldDefinition("kev_date_added", "datetime", "KEV date added", "KEVEnricher")
    ]

    Registry.register_schema("KEVEnricher", fields)

    # Should be retrievable
    result = Registry.get_component_columns("KEVEnricher")
    assert len(result) == 1
    assert result[0].name == "kev_date_added"


def test_get_all_columns():
    """Registry.get_all_columns() returns all registered fields."""
    Registry.clear()

    kev_fields = [
        FieldDefinition("kev_date_added", "datetime", "KEV date", "KEVEnricher")
    ]
    epss_fields = [
        FieldDefinition("epss_probability", "float", "EPSS probability", "EPSSEnricher")
    ]

    Registry.register_schema("KEVEnricher", kev_fields)
    Registry.register_schema("EPSSEnricher", epss_fields)

    all_fields = Registry.get_all_columns()

    assert len(all_fields) == 2
    field_names = {f.name for f in all_fields}
    assert "kev_date_added" in field_names
    assert "epss_probability" in field_names


def test_get_component_columns_nonexistent():
    """Registry.get_component_columns() returns empty list for unknown component."""
    Registry.clear()

    result = Registry.get_component_columns("NonExistent")

    assert result == []
