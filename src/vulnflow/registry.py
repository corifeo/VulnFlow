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
