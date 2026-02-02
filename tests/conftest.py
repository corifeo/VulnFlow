"""
Shared test fixtures for VulnFlow.

Provides:
- fixtures_dir: Path to fixtures directory
- nvd_sample_path: Path to NVD sample JSON
- kev_sample_path: Path to KEV sample CSV
- epss_sample_path: Path to EPSS sample CSV
- tmp_dir: Temporary directory for test outputs
- MockExtractor, MockTransformer, MockLoader: Test doubles for pipeline stages
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

# Add tests directory to path for fixture imports
tests_dir = Path(__file__).parent
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))

if TYPE_CHECKING:
    from vulnflow.dataset import VulnDataset
    from vulnflow.schema import VulnRecord


class MockExtractor:
    """Test double for Extractor protocol."""

    def __init__(self, name: str = "mock_extractor", num_records: int = 3) -> None:
        self.name = name
        self.num_records = num_records
        self.extract_called = False

    def extract(self) -> list[VulnRecord]:
        """Return mock VulnRecords."""
        from vulnflow.schema import VulnRecord

        self.extract_called = True
        return [
            VulnRecord(
                id=uuid4(),
                cve_id=f"CVE-2024-{i:04d}",
                description=f"Mock vulnerability {i} from {self.name}",
                sources=[self.name],
            )
            for i in range(self.num_records)
        ]


class MockTransformer:
    """Test double for Transform protocol."""

    def __init__(self, name: str = "mock_transformer") -> None:
        self.name = name
        self.depends_on: list[str] = []
        self.transform_called = False

    def transform(self, dataset: VulnDataset) -> None:
        """Mark transform as called (no-op for testing)."""
        self.transform_called = True


class MockLoader:
    """Test double for Loader protocol."""

    def __init__(self, name: str = "mock_loader") -> None:
        self.name = name
        self.load_called = False
        self.loaded_dataset: VulnDataset | None = None

    def load(self, dataset: VulnDataset) -> None:
        """Capture dataset for verification."""
        self.load_called = True
        self.loaded_dataset = dataset


@pytest.fixture
def fixtures_dir() -> Path:
    """Path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def nvd_sample_path(fixtures_dir: Path) -> Path:
    """Path to NVD sample JSON file."""
    return fixtures_dir / "nvd_sample.json"


@pytest.fixture
def kev_sample_path(fixtures_dir: Path) -> Path:
    """Path to KEV sample CSV file."""
    return fixtures_dir / "kev_sample.csv"


@pytest.fixture
def epss_sample_path(fixtures_dir: Path) -> Path:
    """Path to EPSS sample CSV file."""
    return fixtures_dir / "epss_sample.csv"


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    """Temporary directory for test outputs."""
    return tmp_path
