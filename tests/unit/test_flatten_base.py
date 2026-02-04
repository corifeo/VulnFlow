"""
BaseFlatten - Abstract base for flatten transforms
Tests: Abstract class behavior, input_column requirement

Layer: Transform/Flatten
Dependencies: transforms.base
"""

from __future__ import annotations

import pytest


def test_base_flatten_is_abstract():
    """BaseFlatten cannot be instantiated directly."""
    from vulnflow.transforms.flatten.base import BaseFlatten

    with pytest.raises(TypeError):
        BaseFlatten()  # type: ignore[abstract]


def test_base_flatten_requires_input_column():
    """Subclass must define input_column."""
    from vulnflow.transforms.flatten.base import BaseFlatten

    class TestFlatten(BaseFlatten):
        name = "test"
        input_column = "test_scores"
        OUTPUT_SCHEMA: list = []

        def transform(self, dataset):  # type: ignore[override]
            pass

    t = TestFlatten()
    assert t.input_column == "test_scores"
