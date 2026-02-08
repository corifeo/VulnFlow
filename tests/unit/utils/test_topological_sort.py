"""
Graph Utilities - Tests for topological sort

Tests: Dependency resolution for transform ordering
Layer: Utility
Dependencies: None
"""

from __future__ import annotations

import pytest

from vulnflow.utils.graph import build_phases, topological_sort


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


# =============================================================================
# build_phases Tests
# =============================================================================


class _T:
    """Minimal transform stub for phase tests."""

    def __init__(self, name: str, depends_on: list[str] | None = None) -> None:
        self.name = name
        self.depends_on = depends_on or []


def test_build_phases_groups_independent_transforms():
    """Independent transforms land in the same phase."""
    phases = build_phases([_T("a"), _T("b"), _T("c")])

    assert len(phases) == 1
    assert {t.name for t in phases[0]} == {"a", "b", "c"}


def test_build_phases_respects_dependencies():
    """Dependent transforms land in later phases."""
    phases = build_phases([_T("c", ["b"]), _T("a"), _T("b", ["a"])])

    assert len(phases) == 3
    assert phases[0][0].name == "a"
    assert phases[1][0].name == "b"
    assert phases[2][0].name == "c"


def test_build_phases_mixed_independent_and_dependent():
    """Mix of independent and dependent transforms grouped correctly."""
    phases = build_phases([
        _T("priority", ["cvd_state"]),
        _T("cvd_state", ["kev", "epss"]),
        _T("kev"),
        _T("epss"),
        _T("cvss_flatten"),
    ])

    assert len(phases) == 3
    assert {t.name for t in phases[0]} == {"kev", "epss", "cvss_flatten"}
    assert phases[1][0].name == "cvd_state"
    assert phases[2][0].name == "priority"


def test_build_phases_detects_circular_dependency():
    """Circular dependency raises ValueError."""
    with pytest.raises(ValueError, match="Circular dependency"):
        build_phases([_T("a", ["b"]), _T("b", ["a"])])


def test_build_phases_handles_missing_dependency():
    """Missing dependency treated as already satisfied (lenient)."""
    phases = build_phases([_T("a", ["nonexistent"]), _T("b", ["a"])])

    # "nonexistent" not in list → assumed satisfied → "a" goes in phase 0
    assert len(phases) == 2
    assert phases[0][0].name == "a"
    assert phases[1][0].name == "b"


def test_build_phases_empty_list():
    """Empty input returns empty phases."""
    assert build_phases([]) == []
