"""
Registry - Plugin registration and lookup
Tests: @register_transform, get_transforms()

Layer: Core plugin system
Dependencies: None
"""

from __future__ import annotations

from vulnflow.registry import clear_registry, get_transforms, register_transform


def test_register_transform_decorator():
    """@register_transform registers transform class."""
    clear_registry()

    @register_transform
    class TestTransform:
        name = "test"
        depends_on: list[str] = []

        def transform(self, dataset):  # type: ignore[no-untyped-def]
            pass

    transforms = get_transforms()
    assert len(transforms) == 1
    assert transforms[0].__name__ == "TestTransform"


def test_get_transforms_returns_list():
    """get_transforms() returns list of registered transforms."""
    clear_registry()

    @register_transform
    class Transform1:
        name = "t1"
        depends_on: list[str] = []

    @register_transform
    class Transform2:
        name = "t2"
        depends_on: list[str] = []

    transforms = get_transforms()
    assert len(transforms) == 2


def test_clear_registry():
    """clear_registry() removes all registrations."""
    clear_registry()

    @register_transform
    class TestTransform:
        name = "test"
        depends_on: list[str] = []

    assert len(get_transforms()) == 1

    clear_registry()
    assert len(get_transforms()) == 0


def test_register_enricher_decorator():
    """@register_enricher registers enricher class."""
    from vulnflow.registry import get_enrichers, register_enricher

    clear_registry()

    @register_enricher
    class TestEnricher:
        name = "test"

        async def enrich(self, dataset):  # type: ignore[no-untyped-def]
            pass

    enrichers = get_enrichers()
    assert len(enrichers) == 1
    assert enrichers[0].__name__ == "TestEnricher"


def test_get_enrichers_returns_list():
    """get_enrichers() returns list of registered enrichers."""
    from vulnflow.registry import get_enrichers, register_enricher

    clear_registry()

    @register_enricher
    class Enricher1:
        name = "e1"

    @register_enricher
    class Enricher2:
        name = "e2"

    enrichers = get_enrichers()
    assert len(enrichers) == 2


def test_registry_register_extractor():
    """Registry.register_extractor() registers extractor class."""
    from vulnflow.registry import Registry

    clear_registry()

    @Registry.register_extractor("test_extractor")
    class TestExtractor:
        name = "test_extractor"
        format = "test"

        def extract(self):  # type: ignore[no-untyped-def]
            pass

    extractor_cls = Registry.get_extractor("test_extractor")
    assert extractor_cls is not None
    assert extractor_cls.__name__ == "TestExtractor"


def test_registry_get_extractor_not_found():
    """Registry.get_extractor() raises KeyError for unknown extractor."""
    import pytest

    from vulnflow.registry import Registry

    clear_registry()

    with pytest.raises(KeyError, match="unknown"):
        Registry.get_extractor("unknown")


def test_registry_list_extractors():
    """Registry.list_extractors() returns all registered extractor names."""
    from vulnflow.registry import Registry

    clear_registry()

    @Registry.register_extractor("extractor1")
    class Extractor1:
        name = "extractor1"
        format = "fmt1"

    @Registry.register_extractor("extractor2")
    class Extractor2:
        name = "extractor2"
        format = "fmt2"

    extractors = Registry.list_extractors()
    assert len(extractors) == 2
    assert "extractor1" in extractors
    assert "extractor2" in extractors


def test_registry_clear_extractors():
    """Registry.clear() clears extractor registrations."""
    from vulnflow.registry import Registry

    clear_registry()

    @Registry.register_extractor("test")
    class TestExtractor:
        name = "test"
        format = "test"

    assert len(Registry.list_extractors()) == 1

    clear_registry()
    assert len(Registry.list_extractors()) == 0
