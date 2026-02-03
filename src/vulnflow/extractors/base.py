"""
Extractor Base - Protocol for data extraction

Provides:
- Extractor: Protocol defining extractor interface
- BaseExtractor: ABC for extractors with format validation

Layer: Extract
Dependencies: None
Used by: nvd, files, sql, memory extractors
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from vulnflow.exceptions import ExtractorError
from vulnflow.utils import json

if TYPE_CHECKING:
    from vulnflow.dataset import VulnDataset

__all__ = ["Extractor", "BaseExtractor"]


@runtime_checkable
class Extractor(Protocol):
    """Protocol for extractors - source data → VulnDataset.

    Extractors are responsible for:
    - Reading source files
    - Parsing source formats
    - Converting to columnar storage
    - Applying domain-specific flattening
    - Returning VulnDataset directly

    Extractors own their conversion logic. VulnDataset should not
    contain extraction or conversion logic.
    """

    name: str

    def extract(self) -> VulnDataset:
        """Extract from source and return VulnDataset.

        Returns:
            VulnDataset with columnar storage populated

        Raises:
            FileNotFoundError: If source file doesn't exist
            ValueError: If source format is invalid
        """
        ...


class BaseExtractor(ABC):
    """Base class for extractors with format validation.

    Provides:
    - Format validation on initialization
    - Helper methods for common operations
    - Abstract extract() method

    Subclasses must:
    - Define format: str class attribute
    - Implement extract() method

    Example:
        >>> class JSONExtractor(BaseExtractor):
        ...     format = "json"
        ...     name = "JSONExtractor"
        ...
        ...     def extract(self) -> VulnDataset:
        ...         # Implementation
        ...         pass
    """

    format: str
    name: str

    def __init__(self) -> None:
        """Initialize extractor and validate format attribute.

        Raises:
            ExtractorError: If format attribute is missing or empty
        """
        if not hasattr(self, "format") or not self.format:
            raise ExtractorError(
                f"{self.__class__.__name__} must define a non-empty 'format' class attribute"
            )

    @abstractmethod
    def extract(self) -> VulnDataset:
        """Extract from source and return VulnDataset.

        Returns:
            VulnDataset with columnar storage populated

        Raises:
            ExtractorError: If extraction fails
        """
        ...

    def _ensure_file_exists(self, path: Path) -> None:
        """Verify file exists, raise ExtractorError if not.

        Args:
            path: Path to verify

        Raises:
            ExtractorError: If file doesn't exist
        """
        if not path.exists():
            raise ExtractorError(f"File not found: {path}")

    def _read_json_safely(self, path: Path) -> Any:
        """Read JSON file with error handling.

        Args:
            path: Path to JSON file

        Returns:
            Parsed JSON data

        Raises:
            ExtractorError: If file cannot be read or parsed
        """
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ExtractorError(f"Invalid JSON in {path}: {e}") from e
        except OSError as e:
            raise ExtractorError(f"Cannot read file {path}: {e}") from e
