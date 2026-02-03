"""
Graph Utilities - Dependency resolution

Provides:
- topological_sort: Sort transforms by dependencies

Layer: Utility
Dependencies: None
Used by: dataset
"""

from __future__ import annotations


def topological_sort(transforms: list[object]) -> list[object]:
    """Sort transforms by dependencies using simple O(n²) algorithm.

    Limitations vs. Kahn's algorithm:
    - No parallel execution groups (can't identify independent transforms)
    - Lenient on missing dependencies (assumes already satisfied)
    - Slower O(n²) vs O(n+e) for large transform lists
    - Good enough: typical usage has <10 transforms

    Args:
        transforms: List of transform instances with name and depends_on attributes

    Returns:
        Sorted list where dependencies come before dependents

    Raises:
        ValueError: If circular dependencies detected
    """
    by_name = {t.name: t for t in transforms}  # type: ignore[attr-defined]
    result: list[object] = []
    remaining = list(transforms)

    while remaining:
        # Find transforms with all dependencies satisfied
        ready = [
            t for t in remaining
            if all(
                dep in [r.name for r in result] or dep not in by_name  # type: ignore[attr-defined]
                for dep in getattr(t, "depends_on", [])
            )
        ]

        if not ready:
            remaining_names = [t.name for t in remaining]  # type: ignore[attr-defined]
            raise ValueError(f"Circular dependency among: {remaining_names}")

        result.extend(ready)
        for t in ready:
            remaining.remove(t)

    return result
