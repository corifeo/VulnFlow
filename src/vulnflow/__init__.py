"""
VulnFlow - ETL Pipeline for Vulnerability Intelligence

Provides:
- VulnDataset: Core container for vulnerability records
- VulnRecord: Dataclass schema for individual records
- VulnPipeline: ETL pipeline orchestrator with composition support
- PipelineResult: Execution results with metadata and error tracking
- Registry: Plugin registration for enrichers and transforms
- Exceptions: VulnFlowError, ExtractorError, EnricherError, TransformError, LoaderError

Layer: Public API
Dependencies: dataset, schema, registry, pipeline, exceptions
Used by: User code
"""

# Import transform modules to trigger registration decorators
import vulnflow.transforms.computed  # noqa: F401
import vulnflow.transforms.enrichers  # noqa: F401
import vulnflow.transforms.inferences  # noqa: F401
from vulnflow.dataset import VulnDataset
from vulnflow.exceptions import (
    EnricherError,
    ExtractorError,
    LoaderError,
    TransformError,
    VulnFlowError,
)
from vulnflow.pipeline import PipelineResult, VulnPipeline
from vulnflow.registry import FieldDefinition, Registry
from vulnflow.schema import VulnRecord

__all__ = [
    "VulnDataset",
    "VulnRecord",
    "VulnPipeline",
    "PipelineResult",
    "Registry",
    "FieldDefinition",
    "VulnFlowError",
    "ExtractorError",
    "EnricherError",
    "TransformError",
    "LoaderError",
]
__version__ = "0.1.0"
