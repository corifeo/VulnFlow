"""
Transforms - Enrichers and computed transforms

Provides:
- Enricher: Protocol for async enrichment (I/O-bound)
- Transform: Protocol for sync transforms (CPU-bound)

Layer: Transform
Dependencies: schema, dataset
Used by: dataset.enrich(), dataset.transform()
"""

from vulnflow.transforms.base import Enricher, Transform

__all__ = ["Enricher", "Transform"]
