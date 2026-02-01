"""
VulnFlow - ETL Pipeline for Vulnerability Intelligence

Provides:
- VulnDataset: Core container for vulnerability records
- VulnRecord: Dataclass schema for individual records
- Registry: Plugin registration for enrichers and transforms

Layer: Public API
Dependencies: dataset, schema, registry
Used by: User code
"""

from vulnflow.dataset import VulnDataset
from vulnflow.registry import Registry
from vulnflow.schema import VulnRecord

__all__ = ["VulnDataset", "VulnRecord", "Registry"]
__version__ = "0.1.0"
