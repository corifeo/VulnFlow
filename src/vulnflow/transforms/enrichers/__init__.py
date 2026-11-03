"""
Enrichers - Async data enrichment modules

Provides:
- KEVEnricher: CISA KEV catalog enrichment (BulkEnricher pattern)
- EPSSEnricher: FIRST EPSS score enrichment (BulkEnricher pattern)
- CWEEnricher: MITRE CWE database enrichment (BulkEnricher pattern)
- CAPECEnricher: MITRE CAPEC attack pattern enrichment (BulkEnricher pattern)
- APIEnricher: Base class for per-CVE API enrichers
- NVDHistoryEnricher: NVD CVE change history enrichment (per-CVE or date-range)

Layer: Transform/Enrichers
Dependencies: base, schema
Used by: dataset.enrich()
"""

from __future__ import annotations

from vulnflow.transforms.enrichers.api_base import APIEnricher
from vulnflow.transforms.enrichers.capec import CAPECEnricher
from vulnflow.transforms.enrichers.cwe import CWEEnricher
from vulnflow.transforms.enrichers.epss import EPSSEnricher
from vulnflow.transforms.enrichers.kev import KEVEnricher
from vulnflow.transforms.enrichers.nvd_history import NVDHistoryEnricher

__all__ = [
    "KEVEnricher",
    "EPSSEnricher",
    "CWEEnricher",
    "CAPECEnricher",
    "APIEnricher",
    "NVDHistoryEnricher",
]
