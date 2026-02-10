"""
Registry - Plugin registration for transforms, extractors, and loaders

Provides:
- Registry: Central class-level registry for transforms, extractors, and loaders
- FieldDefinition: Schema definition for dataset columns
- register_transform: Module-level decorator (uses class name as key)
- get_transforms / clear_registry: Convenience helpers

Registration happens at import time via decorators:
- Transforms: @register_transform(default=True/False) on the class
- Extractors: @Registry.register_extractor("name") on the class
- Loaders:    @Registry.register_loader("name") on the class

Schema tracking is automatic — BaseTransform.__init__() calls
Registry.register_schema() using OUTPUT_SCHEMA from each transform.

Layer: Core
Dependencies: None
Used by: dataset, pipeline, transforms, extractors, loaders, utils.schema
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, TypeVar, overload

if TYPE_CHECKING:
    from vulnflow.extractors.base import BaseExtractor
    from vulnflow.loaders.base import BaseLoader

T = TypeVar("T")


class FieldType(StrEnum):
    """Semantic type enum for database-ready schema mapping.

    FieldDefinition.type accepts both FieldType members and raw strings
    for backward compatibility.
    """

    STR = "str"
    BOOL = "bool"
    INT = "int"
    FLOAT = "float"
    FLOAT32 = "float32"
    UINT8 = "uint8"
    UINT16 = "uint16"
    DATETIME = "datetime"
    UUID = "uuid"
    LIST_STR = "list[str]"


@dataclass
class FieldDefinition:
    """Schema definition for a single field.

    Attributes:
        name: Column name (e.g., "kev_date_added")
        type: Data type — accepts FieldType or raw string (e.g., "datetime", "float", "str")
        description: Human-readable description
        source_component: Which component adds this field (e.g., "KEVEnricher")
        nullable: Whether the column allows NULL/None values
        is_json: Whether the column stores JSON-encoded data
    """

    name: str
    type: str
    description: str
    source_component: str | None = None
    nullable: bool = True
    is_json: bool = False


class Registry:
    """Central registry for transform, extractor, and loader plugins."""

    _transforms: dict[str, tuple[type[Any], bool]] = {}  # name -> (class, is_default)
    _extractors: dict[str, type[BaseExtractor]] = {}  # name -> class
    _loaders: dict[str, type[BaseLoader]] = {}  # name -> class
    # Schema storage with versioning: {component_name: {"version": "X.Y.Z", "fields": [...]}}
    _schemas: dict[str, dict[str, Any]] = {}

    @classmethod
    def register_transform(
        cls, name: str, *, default: bool = False
    ) -> Callable[[type[T]], type[T]]:
        """Decorator to register a transform class."""

        def decorator(transform_cls: type[T]) -> type[T]:
            cls._transforms[name] = (transform_cls, default)
            return transform_cls

        return decorator

    @classmethod
    def register_extractor(cls, name: str) -> Callable[[type[T]], type[T]]:
        """Decorator to register an extractor class."""

        def decorator(extractor_cls: type[T]) -> type[T]:
            cls._extractors[name] = extractor_cls  # type: ignore[assignment]
            return extractor_cls

        return decorator

    @classmethod
    def register_loader(cls, name: str) -> Callable[[type[T]], type[T]]:
        """Decorator to register a loader class."""

        def decorator(loader_cls: type[T]) -> type[T]:
            cls._loaders[name] = loader_cls  # type: ignore[assignment]
            return loader_cls

        return decorator

    @classmethod
    def get_default_transforms(cls) -> list[Any]:
        """Return instances of all default transforms."""
        return [
            transform_cls()
            for transform_cls, is_default in cls._transforms.values()
            if is_default
        ]

    @classmethod
    def get_transform(cls, name: str) -> type[Any]:
        """Get a transform class by name.

        Args:
            name: Transform name

        Returns:
            Transform class

        Raises:
            KeyError: If transform not found
        """
        entry = cls._transforms.get(name)
        if entry is None:
            available = sorted(cls._transforms.keys())
            raise KeyError(
                f"Transform not found: {name}. "
                f"Available: {', '.join(available) if available else 'none'}"
            )
        return entry[0]

    @classmethod
    def get_extractor(cls, name: str) -> type[BaseExtractor]:
        """Get an extractor class by name.

        Args:
            name: Extractor name

        Returns:
            Extractor class

        Raises:
            KeyError: If extractor not found
        """
        if name not in cls._extractors:
            raise KeyError(f"Extractor not found: {name}")
        return cls._extractors[name]

    @classmethod
    def get_loader(cls, name: str) -> type[BaseLoader]:
        """Get a loader class by name.

        Args:
            name: Loader name

        Returns:
            Loader class

        Raises:
            KeyError: If loader not found
        """
        if name not in cls._loaders:
            raise KeyError(f"Loader not found: {name}")
        return cls._loaders[name]

    @classmethod
    def register_schema(
        cls, component_name: str, fields: list[FieldDefinition], version: str = "1.0.0"
    ) -> None:
        """Register schema fields for a component with versioning.

        Args:
            component_name: Name of component (e.g., "KEVEnricher")
            fields: List of FieldDefinition instances
            version: Semantic version string (default "1.0.0")
        """
        cls._schemas[component_name] = {"version": version, "fields": fields}

    @classmethod
    def get_all_columns(cls) -> list[FieldDefinition]:
        """Get all registered columns from all components.

        Returns:
            List of all FieldDefinition instances
        """
        all_fields: list[FieldDefinition] = []
        for schema_data in cls._schemas.values():
            all_fields.extend(schema_data["fields"])
        return all_fields

    @classmethod
    def get_component_columns(cls, component_name: str) -> list[FieldDefinition]:
        """Get columns added by specific component.

        Args:
            component_name: Name of component

        Returns:
            List of FieldDefinition instances for that component
        """
        schema_data = cls._schemas.get(component_name)
        if schema_data is None:
            return []
        fields: list[FieldDefinition] = schema_data["fields"]
        return fields

    @classmethod
    def get_component_schema_version(cls, component_name: str) -> str:
        """Get schema version for a component.

        Args:
            component_name: Name of component

        Returns:
            Version string (e.g., "1.0.0")

        Raises:
            KeyError: If component not found
        """
        if component_name not in cls._schemas:
            raise KeyError(f"Schema not found: {component_name}")
        version: str = cls._schemas[component_name]["version"]
        return version

    @classmethod
    def get_full_schema(cls) -> list[FieldDefinition]:
        """All columns: base (from VulnRecord) first, then transforms.

        Lazily registers base schema on first call via build_base_schema().
        Deduplicates by column name — transform fields override base fields.

        Returns:
            List of FieldDefinition in stable order (base columns first)
        """
        if "_base" not in cls._schemas:
            from vulnflow.utils.schema import build_base_schema

            cls.register_schema("_base", build_base_schema())

        seen: dict[str, FieldDefinition] = {}

        # Base fields first
        base_data = cls._schemas.get("_base")
        if base_data is not None:
            for field_def in base_data["fields"]:
                seen[field_def.name] = field_def

        # Transform fields (override base if same name)
        for component_name, schema_data in cls._schemas.items():
            if component_name == "_base":
                continue
            for field_def in schema_data["fields"]:
                seen[field_def.name] = field_def

        return list(seen.values())

    @classmethod
    def get_schema_for_columns(cls, columns: list[str]) -> list[FieldDefinition]:
        """FieldDefinitions for specific column names.

        Looks up columns in the full schema. Columns not found in the
        registry are returned with inferred type "str" and nullable=True.

        Args:
            columns: Column names to look up

        Returns:
            List of FieldDefinition in the same order as input columns
        """
        schema_map = {fd.name: fd for fd in cls.get_full_schema()}
        result: list[FieldDefinition] = []
        for col in columns:
            if col in schema_map:
                result.append(schema_map[col])
            else:
                result.append(
                    FieldDefinition(
                        name=col,
                        type="str",
                        description=f"Unregistered column: {col}",
                        nullable=True,
                    )
                )
        return result

    @classmethod
    def list_extractors(cls) -> list[str]:
        """Get list of all registered extractor names.

        Returns:
            List of extractor names
        """
        return list(cls._extractors.keys())

    @classmethod
    def list_loaders(cls) -> list[str]:
        """Get list of all registered loader names.

        Returns:
            List of loader names
        """
        return list(cls._loaders.keys())

    @classmethod
    def list_transforms(cls) -> list[str]:
        """Get list of all registered transform names.

        Returns:
            List of transform names
        """
        return list(cls._transforms.keys())

    @classmethod
    def get_default_extractors(cls) -> list[Any]:
        """Return instances of all default extractors.

        Returns:
            List of default extractor instances (currently empty)
        """
        # Extractors don't have default flag yet - return empty list
        return []

    @classmethod
    def get_default_loaders(cls) -> list[Any]:
        """Return instances of all default loaders.

        Returns:
            List of default loader instances (currently empty)
        """
        # Loaders don't have default flag yet - return empty list
        return []

    @classmethod
    def clear(cls) -> None:
        """Clear all registrations (for testing)."""
        cls._transforms = {}
        cls._extractors = {}
        cls._loaders = {}
        cls._schemas = {}


# Module-level convenience functions


@overload
def register_transform(cls: type[T], *, default: bool = False) -> type[T]: ...


@overload
def register_transform(
    cls: None = None, *, default: bool = False
) -> Callable[[type[T]], type[T]]: ...


def register_transform(
    cls: type[T] | None = None, *, default: bool = False
) -> type[T] | Callable[[type[T]], type[T]]:
    """Simple decorator to register a transform class.

    Uses the class name as the registration key.

    Args:
        cls: Transform class to register
        default: If True, include in default transforms

    Returns:
        The same class (for decorator usage)
    """

    def decorator(transform_cls: type[T]) -> type[T]:
        name = getattr(transform_cls, "name", transform_cls.__name__)
        Registry._transforms[name] = (transform_cls, default)
        return transform_cls

    # Support both @register_transform and @register_transform(default=True)
    if cls is None:
        return decorator
    return decorator(cls)


def get_transforms() -> list[type[Any]]:
    """Get list of all registered transform classes.

    Returns:
        List of registered transform classes
    """
    return [cls for cls, _ in Registry._transforms.values()]


def clear_registry() -> None:
    """Clear all registrations (for testing)."""
    Registry.clear()
