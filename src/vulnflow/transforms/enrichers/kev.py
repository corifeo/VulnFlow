"""
KEV Enricher - CISA Known Exploited Vulnerabilities catalog

Provides:
- KEVEnricher: Enriches records with KEV data

Layer: Transform/Enrichers
Dependencies: schema
Used by: dataset.enrich(), pipeline
"""

from __future__ import annotations


class KEVEnricher:
    """Enriches vulnerability records with CISA KEV catalog data.

    Stub implementation for Phase 0. Full implementation in Phase 4.
    """

    name: str = "kev"

    def __init__(self, cache_path: str | None = None) -> None:
        self.cache_path = cache_path
