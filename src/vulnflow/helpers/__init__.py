"""
Helpers - User-facing automation utilities for notebooks

Provides:
- CacheBuilder: One-line NVD vulnerability library construction

Layer: User-facing helpers
Dependencies: pipeline, extractors, loaders
Used by: Notebooks, user scripts

Note: helpers/ is for user-facing automation (imported in notebooks).
      utils/ is for internal library plumbing (imported by vulnflow itself).
"""

from __future__ import annotations

from vulnflow.helpers.builder import CacheBuilder

__all__ = ["CacheBuilder"]
