"""
Topology Resolver - Dependency-based transform ordering

Provides:
- topological_sort: Order transforms by dependencies using Kahn's algorithm
- CyclicDependencyError: Exception for circular dependencies

Layer: Core
Dependencies: None
Used by: dataset.transform(), pipeline.run()
"""

from __future__ import annotations

from typing import Any


class CyclicDependencyError(Exception):
    """Raised when circular dependencies detected in transform graph."""

    pass


def topological_sort(transforms: list[Any]) -> list[Any]:
    """Sort transforms by dependencies using Kahn's algorithm.

    Args:
        transforms: List of transform instances with 'name' and 'depends_on' attributes

    Returns:
        Sorted list where dependencies come before dependents

    Raises:
        CyclicDependencyError: If circular dependencies detected

    Note:
        Stable sort - preserves original order for independent transforms
    """
    # Build name -> transform mapping
    transform_map = {t.name: t for t in transforms}

    # Build dependency graph
    # in_degree[name] = number of unresolved dependencies
    in_degree: dict[str, int] = {}
    # dependents[name] = list of transforms that depend on this one
    dependents: dict[str, list[str]] = {}

    for t in transforms:
        name = t.name
        deps = getattr(t, "depends_on", [])

        # Only count dependencies that are in the current transform list
        # (other dependencies may have been satisfied in previous runs)
        deps_in_list = [d for d in deps if d in transform_map]
        in_degree[name] = len(deps_in_list)

        for dep in deps_in_list:
            if dep not in dependents:
                dependents[dep] = []
            dependents[dep].append(name)

    # Kahn's algorithm: process nodes with no dependencies
    queue = [t.name for t in transforms if in_degree[t.name] == 0]
    result: list[Any] = []

    while queue:
        # Process node with no dependencies
        name = queue.pop(0)
        result.append(transform_map[name])

        # Reduce in-degree for dependents
        for dependent in dependents.get(name, []):
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    # Check for cycles
    if len(result) != len(transforms):
        remaining = [t.name for t in transforms if t not in result]
        raise CyclicDependencyError(f"Cyclic dependency detected among: {remaining}")

    return result
