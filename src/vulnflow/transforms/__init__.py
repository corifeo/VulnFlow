"""
Transforms - Data enrichment and computation

Provides:
- BaseTransform: Root abstract base class
- BaseEnricher: Async I/O-bound transforms
- BaseInference: Sync transforms with confidence
- BaseComputed: Sync pure computation transforms
- Enricher, Transform, Inference: Protocols (type hints)

Layer: Transform
Dependencies: registry, base
Used by: enrichers/, computed/, inferences/
"""

# Import all transform modules to trigger registration decorators
from vulnflow.transforms import computed, enrichers, inferences  # noqa: F401
from vulnflow.transforms.base import (
    BaseComputed,
    BaseEnricher,
    BaseInference,
    BaseTransform,
    Enricher,
    Inference,
    Transform,
)

__all__ = [
    "BaseTransform",
    "BaseEnricher",
    "BaseInference",
    "BaseComputed",
    "Enricher",
    "Transform",
    "Inference",
]
