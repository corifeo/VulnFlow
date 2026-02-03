"""
Loader Base - Protocol for data loading/export

Provides:
- Loader: Protocol defining loader interface
- BaseLoader: ABC for loaders with format validation

Layer: Load
Dependencies: dataset
Used by: files, dataframe, sql loaders
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from vulnflow.exceptions import LoaderError
from vulnflow.utils import json

if TYPE_CHECKING:
    from vulnflow.dataset import VulnDataset

__all__ = ["Loader", "BaseLoader"]


@runtime_checkable
class Loader(Protocol):
    """Protocol for data loaders/exporters.

    Loaders are responsible for:
    - Writing dataset to destination
    - Handling file I/O and format conversion
    - Creating parent directories if needed
    - Returning metadata dict with path and record count

    Loaders should not modify the dataset.
    """

    format: str

    def load(self, dataset: VulnDataset) -> dict[str, Any]:
        """Load/export dataset to destination.

        Returns:
            Metadata dict with loader-specific information
        """
        ...


class BaseLoader(ABC):
    """Base class for loaders with format validation.

    Provides:
    - Format validation on initialization
    - Helper methods for common operations
    - Abstract load() method

    Subclasses must:
    - Define format: str class attribute
    - Implement load() method

    Example:
        >>> class JSONLoader(BaseLoader):
        ...     format = "json"
        ...     name = "JSONLoader"
        ...
        ...     def load(self, dataset: VulnDataset) -> dict[str, Any]:
        ...         # Implementation
        ...         pass
    """

    format: str
    name: str

    def __init__(self) -> None:
        """Initialize loader and validate format attribute.

        Raises:
            LoaderError: If format attribute is missing or empty
        """
        if not hasattr(self, "format") or not self.format:
            raise LoaderError(
                f"{self.__class__.__name__} must define a non-empty 'format' class attribute"
            )

    @abstractmethod
    def load(self, dataset: VulnDataset) -> dict[str, Any]:
        """Load/export dataset to destination.

        Args:
            dataset: VulnDataset to load/export

        Returns:
            Metadata dict with loader-specific information

        Raises:
            LoaderError: If loading fails
        """
        ...

    def _create_parent_dirs(self, path: Path) -> None:
        """Create parent directories if they don't exist.

        Args:
            path: Path to file that needs parent directories

        Raises:
            LoaderError: If directories cannot be created
        """
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise LoaderError(
                f"Cannot create parent directories for {path}: {e}"
            ) from e

    def _write_json_safely(self, path: Path, data: dict[str, Any]) -> None:
        """Write JSON file with error handling.

        Args:
            path: Path to JSON file
            data: Data to write as JSON

        Raises:
            LoaderError: If file cannot be written
        """
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=True, ensure_ascii=False)
        except OSError as e:
            raise LoaderError(f"Cannot write file {path}: {e}") from e
        except (TypeError, ValueError) as e:
            raise LoaderError(f"Cannot serialize data to JSON: {e}") from e
