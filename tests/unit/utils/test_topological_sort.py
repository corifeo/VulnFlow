"""
Graph Utilities - Tests for topological sort

Tests: Dependency resolution for transform ordering
Layer: Utility
Dependencies: None
"""

from __future__ import annotations

import pytest

from vulnflow.utils.graph import topological_sort


def test_topological_sort_orders_by_dependencies():
    """topological_sort orders transforms by depends_on."""

    class Transform1:
        name = "t1"
        depends_on: list[str] = []

    class Transform2:
        name = "t2"
        depends_on = ["t1"]

    class Transform3:
        name = "t3"
        depends_on = ["t2"]

    # Pass in wrong order
    transforms = [Transform3(), Transform1(), Transform2()]
    sorted_transforms = topological_sort(transforms)

    names = [t.name for t in sorted_transforms]
    assert names == ["t1", "t2", "t3"]


def test_topological_sort_handles_independent_transforms():
    """topological_sort handles transforms with no dependencies."""

    class Transform1:
        name = "t1"
        depends_on: list[str] = []

    class Transform2:
        name = "t2"
        depends_on: list[str] = []

    transforms = [Transform2(), Transform1()]
    sorted_transforms = topological_sort(transforms)

    # Both are valid since independent
    assert len(sorted_transforms) == 2
    assert sorted_transforms[0].name in ["t1", "t2"]
    assert sorted_transforms[1].name in ["t1", "t2"]


def test_topological_sort_detects_circular_dependencies():
    """topological_sort raises ValueError on circular dependencies."""

    class Transform1:
        name = "t1"
        depends_on = ["t2"]

    class Transform2:
        name = "t2"
        depends_on = ["t1"]

    transforms = [Transform1(), Transform2()]

    with pytest.raises(ValueError, match="Circular dependency"):
        topological_sort(transforms)


def test_topological_sort_handles_missing_dependencies():
    """topological_sort handles missing dependencies (assumes satisfied)."""

    class Transform1:
        name = "t1"
        depends_on = ["external"]  # Not in list

    class Transform2:
        name = "t2"
        depends_on = ["t1"]

    transforms = [Transform2(), Transform1()]
    sorted_transforms = topological_sort(transforms)

    # Should order t1 before t2 (external assumed satisfied)
    names = [t.name for t in sorted_transforms]
    assert names.index("t1") < names.index("t2")
