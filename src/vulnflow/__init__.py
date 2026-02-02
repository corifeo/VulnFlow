"""
VulnFlow - ETL Pipeline for Vulnerability Intelligence

Provides:
- VulnDataset: Core container for vulnerability records
- VulnRecord: Dataclass schema for individual records
- VulnPipeline: ETL pipeline orchestrator with composition support
- PipelineResult: Execution results with metadata and error tracking
- Registry: Plugin registration for enrichers and transforms

Layer: Public API
Dependencies: dataset, schema, registry, pipeline
Used by: User code
"""

from vulnflow.dataset import VulnDataset
from vulnflow.pipeline import PipelineResult, VulnPipeline
from vulnflow.registry import Registry
from vulnflow.schema import VulnRecord

__all__ = ["VulnDataset", "VulnRecord", "VulnPipeline", "PipelineResult", "Registry"]
__version__ = "0.1.0"
