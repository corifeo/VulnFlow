"""
VulnFlow - ETL Pipeline for Vulnerability Intelligence

Provides:
- VulnDataset: Core container for vulnerability records
- VulnRecord: Dataclass schema for individual records
- VulnPipeline: ETL pipeline orchestrator with composition support
- PipelineResult: Execution results with metadata and error tracking
- Registry: Plugin registration for enrichers and transforms
- Extractor, BaseExtractor: Protocol and base class for extractors
- Loader, BaseLoader: Protocol and base class for loaders
- Exceptions: VulnFlowError, ExtractorError, EnricherError, TransformError, LoaderError, PipelineError

Layer: Public API
Dependencies: dataset, schema, registry, pipeline, exceptions, extractors, loaders
Used by: User code
"""

# Import extractor and transform modules to trigger registration decorators
import vulnflow.extractors.msrc  # noqa: F401
import vulnflow.transforms.computed  # noqa: F401
import vulnflow.transforms.enrichers  # noqa: F401
import vulnflow.transforms.inferences  # noqa: F401
from vulnflow.dataset import VulnDataset
from vulnflow.exceptions import (
    EnricherError,
    ExtractorError,
    LoaderError,
    PipelineError,
    TransformError,
    VulnFlowError,
)
from vulnflow.extractors.base import BaseExtractor, Extractor
from vulnflow.loaders.base import BaseLoader, Loader
from vulnflow.pipeline import PipelineResult, VulnPipeline
from vulnflow.registry import FieldDefinition, FieldType, Registry
from vulnflow.schema import VulnRecord

__all__ = [
    "VulnDataset",
    "VulnRecord",
    "VulnPipeline",
    "PipelineResult",
    "Registry",
    "FieldDefinition",
    "FieldType",
    "Extractor",
    "BaseExtractor",
    "Loader",
    "BaseLoader",
    "VulnFlowError",
    "ExtractorError",
    "EnricherError",
    "TransformError",
    "LoaderError",
    "PipelineError",
]
__version__ = "0.1.0"
