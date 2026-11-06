"""
Transform Base - Base classes and protocols for transforms

Provides:
- BaseTransform: Root abstract base class (enforces patterns)
- BaseEnricher: Async I/O-bound transforms (external data)
- BaseInference: Sync transforms with confidence threshold (fill gaps)
- BaseComputed: Sync pure computation transforms (derive fields)
- Enricher, Transform, Inference: Protocols (for type hints)

Layer: Transform
Dependencies: registry
Used by: enrichers/, computed/, inferences/

Architecture:
  BaseTransform (ABC root)
    ├─ BaseEnricher (async)
    ├─ BaseInference (sync + confidence)
    └─ BaseComputed (sync)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from vulnflow.registry import FieldDefinition, Registry
from vulnflow.utils.stage import StageObservability

if TYPE_CHECKING:
    from vulnflow.dataset import VulnDataset


# ============================================================================
# Abstract Base Classes (for implementation - enforce patterns)
# ============================================================================


class BaseTransform(StageObservability, ABC):
    """Root abstract base class for ALL transforms.

    Enforces:
    - name attribute (class variable)
    - OUTPUT_SCHEMA attribute (list[FieldDefinition])
    - transform() method (abstract)
    - Automatic schema registration with versioning

    Provides:
    - depends_on = [] (default, can override)
    - SCHEMA_VERSION = "1.0.0" (default, can override)
    """

    # Required class attributes (subclass must define)
    name: str
    OUTPUT_SCHEMA: list[FieldDefinition]

    # Optional class attributes (subclass can override)
    depends_on: list[str] = []
    SCHEMA_VERSION: str = "1.0.0"

    @property
    def label(self) -> str:
        """Display name with parens to indicate callable stage."""
        return f"{self.name}()"

    def __init__(self) -> None:
        """Initialize transform and register schema.

        Validates:
        - name is defined
        - OUTPUT_SCHEMA is defined and is list[FieldDefinition]
        - depends_on is explicitly declared and is a list

        Side effects:
        - Registers schema with Registry
        """
        self._init_observability()
        # Validate required attributes
        if not hasattr(self.__class__, "name"):
            raise TypeError(
                f"{self.__class__.__name__} must define 'name' class attribute"
            )

        if not hasattr(self.__class__, "OUTPUT_SCHEMA"):
            raise AttributeError(
                f"{self.__class__.__name__} must define 'OUTPUT_SCHEMA' class attribute"
            )

        # Validate OUTPUT_SCHEMA format
        if not isinstance(self.OUTPUT_SCHEMA, list):
            raise TypeError(f"{self.__class__.__name__}.OUTPUT_SCHEMA must be a list")

        for item in self.OUTPUT_SCHEMA:
            if not isinstance(item, FieldDefinition):
                raise TypeError(
                    f"{self.__class__.__name__}.OUTPUT_SCHEMA must contain FieldDefinition instances, "
                    f"got {type(item).__name__}"
                )

        # Validate depends_on is explicitly declared
        if "depends_on" not in self.__class__.__dict__:
            raise AttributeError(
                f"{self.__class__.__name__} must explicitly declare 'depends_on' class attribute. "
                f"Use 'depends_on: list[str] = []' if no dependencies."
            )

        # Validate depends_on type
        if not isinstance(self.depends_on, list):
            raise TypeError(
                f"{self.__class__.__name__}.depends_on must be a list, "
                f"got {type(self.depends_on).__name__}"
            )

        # Automatically register schema with version
        version = getattr(self.__class__, "SCHEMA_VERSION", "1.0.0")
        Registry.register_schema(self.name, self.OUTPUT_SCHEMA, version)

    @abstractmethod
    def transform(self, dataset: VulnDataset) -> None:
        """Transform dataset (modifies in place).

        Args:
            dataset: VulnDataset to transform
        """
        ...


class BaseEnricher(BaseTransform):
    """Abstract base class for async enrichers (I/O-bound).

    Enrichers fetch external data and add new columns.
    Examples: KEVEnricher, EPSSEnricher
    """

    @abstractmethod
    async def transform(self, dataset: VulnDataset) -> None:  # type: ignore[override]
        """Enrich dataset with external data (async).

        Args:
            dataset: VulnDataset to enrich (modifies in place)
        """
        ...


class BaseInference(BaseTransform):
    """Abstract base class for inference transforms (sync with confidence).

    Inferences fill gaps in data using heuristics.
    Examples: EventInference, TagInference

    Standard pattern:
    - TIER_THRESHOLDS: Maps tier number to minimum confidence
    - confidence_threshold: Gates which tiers run (threshold <= tier value)
    - overwrite: If True, re-infer even if output columns already populated
    - tiers: Explicit tier list (overrides threshold gating)
    - _should_run(tier): Check if a tier should execute
    """

    # Optional: default confidence threshold
    confidence_threshold: float = 0.7

    # Subclass defines: {tier_number: minimum_confidence}
    TIER_THRESHOLDS: dict[int, float] = {}

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        overwrite: bool = False,
        tiers: list[int] | None = None,
    ) -> None:
        """Initialize inference with standard parameters.

        Args:
            confidence_threshold: Minimum confidence to run a tier.
            overwrite: Replace existing output values.
            tiers: Explicit tier list (overrides threshold gating).
        """
        super().__init__()
        self.confidence_threshold = confidence_threshold
        self.overwrite = overwrite
        self.tiers = tiers

    def _should_run(self, tier: int) -> bool:
        """Check if a tier should run based on configuration.

        If explicit tiers list is set, checks membership.
        Otherwise, checks confidence_threshold <= tier threshold.
        """
        if self.tiers is not None:
            return tier in self.tiers
        return self.confidence_threshold <= self.TIER_THRESHOLDS.get(tier, 0.0)

    @abstractmethod
    def transform(self, dataset: VulnDataset) -> None:
        """Infer missing fields where confidence >= threshold (sync).

        Args:
            dataset: VulnDataset to infer (modifies in place)
        """
        ...


class BaseComputed(BaseTransform):
    """Abstract base class for computed transforms (sync pure computation).

    Computed transforms derive new fields from existing data.
    Examples: CVDStateTransform, DesiderataTransform
    """

    @abstractmethod
    def transform(self, dataset: VulnDataset) -> None:
        """Compute derived fields (sync).

        Args:
            dataset: VulnDataset to compute (modifies in place)
        """
        ...


# ============================================================================
# Protocols (for type hints - structural typing)
# ============================================================================


@runtime_checkable
class Enricher(Protocol):
    """Protocol for async enrichers that fetch external data.

    Note: For implementation, inherit from BaseEnricher (ABC).
    This protocol is for type hints only.
    """

    name: str

    async def transform(self, dataset: VulnDataset) -> None:
        """Enrich dataset with external data (async, I/O-bound)."""
        ...


@runtime_checkable
class Transform(Protocol):
    """Protocol for sync transforms that compute derived fields.

    Note: For implementation, inherit from BaseComputed (ABC).
    This protocol is for type hints only.
    """

    name: str
    depends_on: list[str]

    def transform(self, dataset: VulnDataset) -> None:
        """Transform dataset by computing derived fields (sync, CPU-bound)."""
        ...


@runtime_checkable
class Inference(Protocol):
    """Protocol for inference transforms that fill gaps with heuristics.

    Note: For implementation, inherit from BaseInference (ABC).
    This protocol is for type hints only.
    """

    name: str
    confidence_threshold: float
    depends_on: list[str]

    def transform(self, dataset: VulnDataset) -> None:
        """Infer missing fields where confidence >= threshold."""
        ...
