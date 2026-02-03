"""
Schema Registry - Track component output schemas

Provides:
- SchemaRegistry: Central registry for component output field definitions
- FieldDefinition: Describes a single output field

Layer: Core
Dependencies: None
Used by: transforms, enrichers, loaders
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FieldDefinition:
    """Definition of a single output field."""

    name: str
    type: str  # "str", "int", "float", "datetime", "bool", etc.
    source: str  # Component that provides this field
    description: str = ""


class SchemaRegistry:
    """Registry for tracking which components provide which fields.

    Usage:
        registry = SchemaRegistry()
        registry.register_schema("KEVEnricher", "1.0.0", [
            FieldDefinition("kev_date_added", "datetime", "KEVEnricher"),
        ])
        fields = registry.get_registered_fields()
    """

    def __init__(self) -> None:
        """Initialize empty schema registry."""
        # component_name -> (version, fields)
        self._schemas: dict[str, tuple[str, list[FieldDefinition]]] = {}
        # field_name -> FieldDefinition
        self._fields: dict[str, FieldDefinition] = {}

    def register_schema(
        self, component_name: str, version: str, fields: list[FieldDefinition]
    ) -> None:
        """Register a component's output schema.

        Args:
            component_name: Name of component (e.g., "KEVEnricher")
            version: Schema version (e.g., "1.0.0")
            fields: List of field definitions
        """
        self._schemas[component_name] = (version, fields)

        # Add to field index
        for field in fields:
            self._fields[field.name] = field

    def get_registered_fields(self) -> dict[str, FieldDefinition]:
        """Get all registered fields.

        Returns:
            Dict mapping field name to FieldDefinition
        """
        return self._fields.copy()

    def get_field_providers(self) -> dict[str, str]:
        """Get mapping of field name to component that provides it.

        Returns:
            Dict mapping field name to component name
        """
        return {name: field.source for name, field in self._fields.items()}

    def clear(self) -> None:
        """Clear all registrations (for testing)."""
        self._schemas.clear()
        self._fields.clear()


# Global singleton instance
_global_registry = SchemaRegistry()


def get_global_registry() -> SchemaRegistry:
    """Get the global schema registry instance."""
    return _global_registry
