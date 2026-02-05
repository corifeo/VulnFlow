"""
Transforms - Data enrichment and computation

Provides:
- BaseTransform: Root abstract base class
- BaseEnricher: Async I/O-bound transforms
- BaseInference: Sync transforms with confidence
- BaseComputed: Sync pure computation transforms
- Enricher, Transform, Inference: Protocols (type hints)
- Flatten transforms: CVSSFlattenTransform, EPSSFlattenTransform, etc.

Layer: Transform
Dependencies: registry, base
Used by: enrichers/, computed/, inferences/, flatten/
"""

# Import all transform modules to trigger registration decorators
from vulnflow.transforms import computed, enrichers, flatten, inferences  # noqa: F401
from vulnflow.transforms.base import (
    BaseComputed,
    BaseEnricher,
    BaseInference,
    BaseTransform,
    Enricher,
    Inference,
    Transform,
)
from vulnflow.transforms.flatten import (
    BaseFlatten,
    CPEFlattenTransform,
    CVSSFlattenTransform,
    CWEFlattenTransform,
    EPSSFlattenTransform,
)

__all__ = [
    "BaseTransform",
    "BaseEnricher",
    "BaseInference",
    "BaseComputed",
    "BaseFlatten",
    "Enricher",
    "Transform",
    "Inference",
    "CVSSFlattenTransform",
    "EPSSFlattenTransform",
    "CPEFlattenTransform",
    "CWEFlattenTransform",
]
