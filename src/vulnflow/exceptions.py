"""
Exceptions - Domain-specific error types

Hierarchy:
- VulnFlowError (base)
  - ExtractorError: Extraction failures (always fatal)
  - EnricherError: Enrichment failures (strict/lenient modes)
  - TransformError: Transform failures (strict/lenient modes)
  - LoaderError: Load failures (always fatal)
  - PipelineError: Pipeline configuration or orchestration errors

Layer: Core infrastructure
"""

from __future__ import annotations


class VulnFlowError(Exception):
    """Base exception for all VulnFlow errors."""

    pass


class ExtractorError(VulnFlowError):
    """Extraction stage failure (always fatal).

    Raised when an extractor cannot read or parse source data. Examples:
    - File not found or unreadable
    - Invalid JSON/Parquet format
    - Missing required fields in source data
    - Unsupported schema version

    Extractors MUST raise this exception immediately on failure.
    Do not catch and continue - extraction failures are always fatal.

    Example:
        >>> if not path.exists():
        ...     raise ExtractorError(f"File not found: {path}")
    """

    pass


class EnricherError(VulnFlowError):
    """Enrichment stage failure (strict/lenient configurable)."""

    pass


class TransformError(VulnFlowError):
    """Transform stage failure (strict/lenient configurable)."""

    pass


class LoaderError(VulnFlowError):
    """Load stage failure (always fatal).

    Raised when a loader cannot write output data. Examples:
    - Permission denied writing file
    - Disk full or I/O error
    - Invalid data format for target
    - Missing parent directory

    Loaders MUST raise this exception immediately on failure.
    Do not catch and continue - load failures risk data loss.

    Example:
        >>> if not parent_dir.exists():
        ...     raise LoaderError(f"Parent directory does not exist: {parent_dir}")
    """

    pass


class PipelineError(VulnFlowError):
    """Pipeline configuration or orchestration error.

    Raised when pipeline structure is invalid for the requested operation.
    Examples:
    - Parallel execution with mismatched extractor/loader counts
    - Parallel execution with transforms (not supported in Pattern 1)

    Example:
        >>> if len(extractors) != len(loaders):
        ...     raise PipelineError("Parallel requires equal extractors and loaders")
    """

    pass
