"""
Exceptions - Domain-specific error types

Hierarchy:
- VulnFlowError (base)
  - ExtractorError: Extraction failures (always fatal)
  - EnricherError: Enrichment failures (strict/lenient modes)
  - TransformError: Transform failures (strict/lenient modes)
  - LoaderError: Load failures (always fatal)

Layer: Core infrastructure
"""

from __future__ import annotations


class VulnFlowError(Exception):
    """Base exception for all VulnFlow errors."""

    pass


class ExtractorError(VulnFlowError):
    """Extraction stage failure (always fatal)."""

    pass


class EnricherError(VulnFlowError):
    """Enrichment stage failure (strict/lenient configurable)."""

    pass


class TransformError(VulnFlowError):
    """Transform stage failure (strict/lenient configurable)."""

    pass


class LoaderError(VulnFlowError):
    """Load stage failure (always fatal)."""

    pass
