"""
Registry - Plugin registration for enrichers and transforms

Provides:
- Registry: Central registry for enricher and transform plugins
- FieldDefinition: Schema definition for dataset fields

Layer: Core
Dependencies: None
Used by: dataset, transforms, enrichers
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar, overload

if TYPE_CHECKING:
    from vulnflow.extractors.base import BaseExtractor
    from vulnflow.loaders.base import BaseLoader

T = TypeVar("T")


@dataclass
class FieldDefinition:
    """Schema definition for a single field.

    Attributes:
        name: Column name (e.g., "kev_date_added")
        type: Data type (e.g., "datetime", "float", "str")
        description: Human-readable description
        source_component: Which component adds this field (e.g., "KEVEnricher")
    """

    name: str
    type: str
    description: str
    source_component: str | None = None


class Registry:
    """Central registry for enricher and transform plugins."""

    _enrichers: dict[str, tuple[type[Any], bool]] = {}  # name -> (class, is_default)
    _transforms: dict[str, tuple[type[Any], bool]] = {}  # name -> (class, is_default)

    _extractors: dict[str, type[BaseExtractor]] = {}  # name -> class
    _loaders: dict[str, type[BaseLoader]] = {}  # name -> class
    _schemas: dict[str, list[FieldDefinition]] = {}  # component_name -> fields

    @classmethod
    def register_enricher(
        cls, name: str, *, default: bool = False
    ) -> Callable[[type[T]], type[T]]:
        """Decorator to register an enricher class."""

        def decorator(enricher_cls: type[T]) -> type[T]:
            cls._enrichers[name] = (enricher_cls, default)
            return enricher_cls

        return decorator

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
    def get_default_enrichers(cls) -> list[Any]:
        """Return instances of all default enrichers."""
        return [
            enricher_cls()
            for enricher_cls, is_default in cls._enrichers.values()
            if is_default
        ]

    @classmethod
    def get_default_transforms(cls) -> list[Any]:
        """Return instances of all default transforms."""
        return [
            transform_cls()
            for transform_cls, is_default in cls._transforms.values()
            if is_default
        ]

    @classmethod
    def get_enricher(cls, name: str) -> type[Any] | None:
        """Get an enricher class by name."""
        entry = cls._enrichers.get(name)
        return entry[0] if entry else None

    @classmethod
    def get_transform(cls, name: str) -> type[Any] | None:
        """Get a transform class by name."""
        entry = cls._transforms.get(name)
        return entry[0] if entry else None

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
    def register_schema(cls, component_name: str, fields: list[FieldDefinition]) -> None:
        """Register schema fields for a component.

        Args:
            component_name: Name of component (e.g., "KEVEnricher")
            fields: List of FieldDefinition instances
        """
        cls._schemas[component_name] = fields

    @classmethod
    def get_all_columns(cls) -> list[FieldDefinition]:
        """Get all registered columns from all components.

        Returns:
            List of all FieldDefinition instances
        """
        all_fields = []
        for component_fields in cls._schemas.values():
            all_fields.extend(component_fields)
        return all_fields

    @classmethod
    def get_component_columns(cls, component_name: str) -> list[FieldDefinition]:
        """Get columns added by specific component.

        Args:
            component_name: Name of component

        Returns:
            List of FieldDefinition instances for that component
        """
        return cls._schemas.get(component_name, [])

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
    def clear(cls) -> None:
        """Clear all registrations (for testing)."""
        cls._enrichers = {}
        cls._transforms = {}
        cls._extractors = {}
        cls._loaders = {}
        cls._schemas = {}


# Module-level convenience functions


@overload
def register_transform(cls: type[T], *, default: bool = False) -> type[T]: ...


@overload
def register_transform(cls: None = None, *, default: bool = False) -> Callable[[type[T]], type[T]]: ...


def register_transform(cls: type[T] | None = None, *, default: bool = False) -> type[T] | Callable[[type[T]], type[T]]:
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


@overload
def register_enricher(cls: type[T], *, default: bool = False) -> type[T]: ...


@overload
def register_enricher(cls: None = None, *, default: bool = False) -> Callable[[type[T]], type[T]]: ...


def register_enricher(cls: type[T] | None = None, *, default: bool = False) -> type[T] | Callable[[type[T]], type[T]]:
    """Simple decorator to register an enricher class.

    Uses the class name as the registration key.

    Args:
        cls: Enricher class to register
        default: If True, include in default enrichers

    Returns:
        The same class (for decorator usage)
    """
    def decorator(enricher_cls: type[T]) -> type[T]:
        name = getattr(enricher_cls, "name", enricher_cls.__name__)
        Registry._enrichers[name] = (enricher_cls, default)
        return enricher_cls

    # Support both @register_enricher and @register_enricher(default=True)
    if cls is None:
        return decorator
    return decorator(cls)


def get_enrichers() -> list[type[Any]]:
    """Get list of all registered enricher classes.

    Returns:
        List of registered enricher classes
    """
    return [cls for cls, _ in Registry._enrichers.values()]


def clear_registry() -> None:
    """Clear all registrations (for testing)."""
    Registry.clear()
