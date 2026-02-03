"""
Topology Resolver - Dependency-based execution ordering
Tests: topological_sort(), cycle detection

Layer: Core
Dependencies: None
"""

from __future__ import annotations

import pytest
from vulnflow.topology import CyclicDependencyError, topological_sort


def test_topological_sort_orders_by_dependencies():
    """topological_sort returns transforms in dependency order."""

    class Transform1:
        name = "t1"
        depends_on: list[str] = []

    class Transform2:
        name = "t2"
        depends_on = ["t1"]  # Depends on t1

    class Transform3:
        name = "t3"
        depends_on = ["t2"]  # Depends on t2

    transforms = [Transform3(), Transform2(), Transform1()]  # Wrong order

    sorted_transforms = topological_sort(transforms)

    # Should be reordered: t1, t2, t3
    assert [t.name for t in sorted_transforms] == ["t1", "t2", "t3"]


def test_topological_sort_handles_independent_transforms():
    """topological_sort preserves order for independent transforms."""

    class Transform1:
        name = "t1"
        depends_on: list[str] = []

    class Transform2:
        name = "t2"
        depends_on: list[str] = []

    transforms = [Transform1(), Transform2()]
    sorted_transforms = topological_sort(transforms)

    # Order should be preserved (stable sort)
    assert [t.name for t in sorted_transforms] == ["t1", "t2"]


def test_topological_sort_detects_cycles():
    """topological_sort raises CyclicDependencyError for circular dependencies."""

    class Transform1:
        name = "t1"
        depends_on = ["t2"]

    class Transform2:
        name = "t2"
        depends_on = ["t1"]  # Cycle!

    transforms = [Transform1(), Transform2()]

    with pytest.raises(CyclicDependencyError, match="Cyclic dependency detected"):
        topological_sort(transforms)


def test_topological_sort_handles_diamond_dependencies():
    """topological_sort handles diamond dependency graph."""

    class Transform1:
        name = "t1"
        depends_on: list[str] = []

    class Transform2:
        name = "t2"
        depends_on = ["t1"]

    class Transform3:
        name = "t3"
        depends_on = ["t1"]

    class Transform4:
        name = "t4"
        depends_on = ["t2", "t3"]  # Depends on both t2 and t3

    transforms = [Transform4(), Transform3(), Transform2(), Transform1()]
    sorted_transforms = topological_sort(transforms)

    names = [t.name for t in sorted_transforms]

    # t1 must come first, t4 must come last
    assert names[0] == "t1"
    assert names[-1] == "t4"
    # t2 and t3 must come before t4
    assert names.index("t2") < names.index("t4")
    assert names.index("t3") < names.index("t4")
