"""
Registry - Plugin registration for enrichers and transforms

Provides:
- Registry: Central registry for enricher and transform plugins

Layer: Core
Dependencies: None
Used by: dataset, transforms, enrichers
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    pass

T = TypeVar("T")


class Registry:
    """Central registry for enricher and transform plugins."""

    _enrichers: dict[str, tuple[type[Any], bool]] = {}  # name -> (class, is_default)
    _transforms: dict[str, tuple[type[Any], bool]] = {}  # name -> (class, is_default)

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
    def clear(cls) -> None:
        """Clear all registrations (for testing)."""
        cls._enrichers = {}
        cls._transforms = {}


# Module-level convenience functions


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
