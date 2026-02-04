"""
Default Registry - Integration tests
Tests: KEV, EPSS, CVDState, EventInference registered as defaults

Layer: Integration
Dependencies: All transforms
"""

from __future__ import annotations

# Import transforms to trigger registration decorators
import vulnflow.transforms.computed  # noqa: F401
import vulnflow.transforms.enrichers  # noqa: F401
import vulnflow.transforms.inferences  # noqa: F401
from vulnflow.registry import Registry


def test_default_transforms_registered():
    """All default transforms registered (KEV, EPSS, EventInference, CVDState)."""
    defaults = Registry.get_default_transforms()

    # Should have at least 4 default transforms
    # (CVDDesiderataTransform was consolidated into CVDStateTransform)
    assert len(defaults) >= 4, f"Expected at least 4 defaults, got {len(defaults)}"

    transform_names = [t.__class__.__name__ for t in defaults]

    # Enrichers (async I/O-bound)
    assert "KEVEnricher" in transform_names
    assert "EPSSEnricher" in transform_names

    # Inferences (sync with confidence)
    assert "EventInference" in transform_names

    # Computed (sync pure computation)
    # CVDStateTransform now computes all 19 CVD fields (including desiderata)
    assert "CVDStateTransform" in transform_names


def test_default_transforms_instantiable():
    """Default transforms can be instantiated without arguments."""
    defaults = Registry.get_default_transforms()

    # All defaults should be instances, not classes
    for transform in defaults:
        assert not isinstance(
            transform, type
        ), f"{transform} should be an instance, not a class"
