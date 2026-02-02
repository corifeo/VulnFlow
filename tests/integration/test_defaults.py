"""
Default Registry - Integration tests
Tests: KEV, EPSS, CVDState, Desiderata, EventInference registered as defaults

Layer: Integration
Dependencies: All enrichers and transforms
"""

from __future__ import annotations

# Import transforms to trigger registration decorators
import vulnflow.transforms.computed  # noqa: F401
import vulnflow.transforms.enrichers  # noqa: F401
import vulnflow.transforms.inferences  # noqa: F401
from vulnflow.registry import Registry


def test_default_enrichers_registered():
    """KEV and EPSS registered as default enrichers."""
    defaults = Registry.get_default_enrichers()

    # Should have 2 default enrichers
    assert len(defaults) >= 2, "Expected KEV and EPSS as defaults"

    enricher_names = [e.__class__.__name__ for e in defaults]
    assert "KEVEnricher" in enricher_names
    assert "EPSSEnricher" in enricher_names


def test_default_transforms_registered():
    """CVDState, Desiderata, and EventInference registered as default transforms."""
    defaults = Registry.get_default_transforms()

    # Should have at least 3 default transforms
    assert len(defaults) >= 3, "Expected EventInference, CVDState, and Desiderata as defaults"

    transform_names = [t.__class__.__name__ for t in defaults]
    assert "EventInference" in transform_names
    assert "CVDStateTransform" in transform_names
    assert "CVDDesiderataTransform" in transform_names


def test_default_enrichers_instantiable():
    """Default enrichers can be instantiated without arguments."""
    defaults = Registry.get_default_enrichers()

    # All defaults should be instances, not classes
    for enricher in defaults:
        assert not isinstance(enricher, type), f"{enricher} should be an instance, not a class"


def test_default_transforms_instantiable():
    """Default transforms can be instantiated without arguments."""
    defaults = Registry.get_default_transforms()

    # All defaults should be instances, not classes
    for transform in defaults:
        assert not isinstance(transform, type), f"{transform} should be an instance, not a class"
