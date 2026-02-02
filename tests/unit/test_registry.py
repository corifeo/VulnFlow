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
