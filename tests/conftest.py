"""
Shared test fixtures for VulnFlow.

Provides:
- fixtures_dir: Path to fixtures directory
- nvd_sample_path: Path to NVD sample JSON
- kev_sample_path: Path to KEV sample CSV
- epss_sample_path: Path to EPSS sample CSV
- tmp_dir: Temporary directory for test outputs
"""

from pathlib import Path

import pytest


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
