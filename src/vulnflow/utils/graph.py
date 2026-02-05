"""Graph utilities - Dependency resolution.

Provides:
- topological_sort: Sort transforms by dependencies using O(n²) algorithm

Layer: Utility
Dependencies: None
Used by: dataset, pipeline

Implementation Notes:
    Uses simple O(n²) greedy algorithm instead of Kahn's algorithm.
    Trade-offs:
    - Slower for large graphs (O(n²) vs O(n+e))
    - Cannot identify independent transforms for parallel execution
    - Lenient on missing dependencies (assumes satisfied)
    - Good enough: typical usage has <10 transforms
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class HasDependencies(Protocol):
    """Protocol for objects with name and depends_on attributes.

    Used by topological_sort to identify dependency relationships.

    Attributes:
        name: Unique identifier for this transform
        depends_on: List of transform names this depends on
    """

    name: str
    depends_on: list[str]


def topological_sort(transforms: list[Any]) -> list[Any]:
    """Sort transforms by dependencies using simple O(n²) algorithm.

    Limitations vs. Kahn's algorithm:
    - No parallel execution groups (can't identify independent transforms)
    - Lenient on missing dependencies (assumes already satisfied)
    - Slower O(n²) vs O(n+e) for large transform lists
    - Good enough: typical usage has <10 transforms

    Args:
        transforms: List of objects with 'name' and 'depends_on' attributes.
                   Missing depends_on attribute is treated as empty list.

    Returns:
        Sorted list where dependencies come before dependents.
        Order is stable for transforms with no dependencies.

    Raises:
        ValueError: If circular dependencies detected. Error message includes:
                   - Remaining transform names that couldn't be sorted
                   - Unsatisfied dependencies for each remaining transform

    Examples:
        >>> class Transform:
        ...     def __init__(self, name, depends_on=None):
        ...         self.name = name
        ...         self.depends_on = depends_on or []
        >>> a = Transform("a")
        >>> b = Transform("b", depends_on=["a"])
        >>> c = Transform("c", depends_on=["b"])
        >>> sorted_transforms = topological_sort([c, a, b])
        >>> [t.name for t in sorted_transforms]
        ['a', 'b', 'c']
    """
    by_name = {t.name: t for t in transforms}
    result: list[Any] = []
    remaining = list(transforms)

    while remaining:
        # Find transforms with all dependencies satisfied
        ready = [
            t
            for t in remaining
            if all(
                dep in [r.name for r in result] or dep not in by_name
                for dep in getattr(t, "depends_on", [])
            )
        ]

        if not ready:
            # Build detailed error message showing unsatisfied dependencies
            satisfied = {r.name for r in result}
            dependency_info = []

            for t in remaining:
                deps = getattr(t, "depends_on", [])
                # Find dependencies that exist but aren't satisfied yet
                unsatisfied = [
                    dep for dep in deps if dep in by_name and dep not in satisfied
                ]
                if unsatisfied:
                    dependency_info.append(f"{t.name} waits for: {unsatisfied}")

            remaining_names = [t.name for t in remaining]

            msg = f"Circular dependency detected among: {remaining_names}"
            if dependency_info:
                msg += "\nUnsatisfied dependencies:\n  " + "\n  ".join(dependency_info)

            raise ValueError(msg)

        result.extend(ready)
        for t in ready:
            remaining.remove(t)

    return result
