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
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import numpy as np
import pytest

# Add tests directory to path for fixture imports
tests_dir = Path(__file__).parent
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))

if TYPE_CHECKING:
    from vulnflow.dataset import VulnDataset
    from vulnflow.schema import VulnRecord


def dataset_from_records(records: list[VulnRecord]) -> VulnDataset:  # type: ignore[name-defined]
    """Helper to create VulnDataset from list of VulnRecord (for testing).

    This is a test helper that mimics the old _from_records behavior.
    In production code, use extractors instead.

    Args:
        records: List of VulnRecord instances

    Returns:
        VulnDataset with columnar storage
    """
    import json
    from dataclasses import fields, is_dataclass

    import numpy as np

    from vulnflow.dataset import VulnDataset
    from vulnflow.schema import CPEEntry, CVSSv3Entry, CVSSv4Entry, CWEEntry, VulnRecord

    # Create JSON encoder once for reuse
    json_encoder = json.JSONEncoder(default=str)

    def dataclass_to_dict(obj: Any) -> Any:
        """Fast dataclass to dict conversion without asdict overhead."""
        if obj is None:
            return None
        if is_dataclass(obj) and not isinstance(obj, type):
            return {
                f.name: getattr(obj, f.name) for f in obj.__dataclass_fields__.values()
            }
        return obj

    def encode_nested(value: Any) -> str:
        """Encode nested structure as JSON string."""
        if value is None:
            return "null"
        if isinstance(value, list):
            if not value:
                return "[]"
            items = [dataclass_to_dict(item) for item in value]
            return json_encoder.encode(items)
        if is_dataclass(value) and not isinstance(value, type):
            return json_encoder.encode(dataclass_to_dict(value))
        return json_encoder.encode(value)

    def flatten_cvss(cvss_scores: list[CVSSv3Entry | CVSSv4Entry]) -> dict[str, float | str | None]:
        """Flatten list[CVSSv3Entry | CVSSv4Entry] to scalar columns."""
        if not cvss_scores:
            return {
                "cvss_base_score": None,
                "cvss_base_severity": None,
                "cvss_version": None,
                "cvss_vector_string": None,
                "cvss_source": None,
            }
        source_priority = {"nvd": 0, "redhat": 1, "microsoft": 2}
        selected = min(cvss_scores, key=lambda e: source_priority.get(e.source, 999))
        return {
            "cvss_base_score": selected.base_score,
            "cvss_base_severity": selected.base_severity,
            "cvss_version": selected.version,
            "cvss_vector_string": selected.vector_string,
            "cvss_source": selected.source,
        }

    def flatten_cwe(cwes: list[CWEEntry]) -> dict[str, str | None]:
        """Flatten list[CWEEntry] to scalar columns (first entry)."""
        if not cwes:
            return {"primary_cwe_id": None, "primary_cwe_name": None}
        first = cwes[0]
        return {"primary_cwe_id": first.id, "primary_cwe_name": first.name}

    def flatten_cpe(cpes: list[CPEEntry]) -> dict[str, str | None]:
        """Flatten list[CPEEntry] to scalar columns (first entry)."""
        if not cpes:
            return {
                "primary_vendor": None,
                "primary_product": None,
                "primary_version": None,
            }
        first = cpes[0]
        return {
            "primary_vendor": first.vendor,
            "primary_product": first.product,
            "primary_version": first.version,
        }

    dataset = VulnDataset()
    if not records:
        return dataset

    n = len(records)
    record_cols = [f.name for f in fields(VulnRecord)]
    flattened_cols = [
        "cvss_base_score",
        "cvss_base_severity",
        "cvss_version",
        "cvss_vector_string",
        "cvss_source",
        "primary_cwe_id",
        "primary_cwe_name",
        "primary_vendor",
        "primary_product",
        "primary_version",
    ]
    all_columns = record_cols + flattened_cols
    arrays = {col: np.empty(n, dtype=object) for col in all_columns}

    for i, r in enumerate(records):
        arrays["id"][i] = str(r.id)
        arrays["cve_id"][i] = r.cve_id
        arrays["aliases"][i] = json.dumps(r.aliases)
        arrays["title"][i] = r.title
        arrays["description"][i] = r.description
        arrays["nvd_published_at"][i] = r.nvd_published_at
        arrays["nvd_modified_at"][i] = r.nvd_modified_at
        arrays["cvss_scores"][i] = encode_nested(r.cvss_scores)
        arrays["epss_scores"][i] = encode_nested(r.epss_scores)
        flattened = flatten_cvss(r.cvss_scores)
        arrays["cvss_base_score"][i] = flattened["cvss_base_score"]
        arrays["cvss_base_severity"][i] = flattened["cvss_base_severity"]
        arrays["cvss_version"][i] = flattened["cvss_version"]
        arrays["cvss_vector_string"][i] = flattened["cvss_vector_string"]
        arrays["cvss_source"][i] = flattened["cvss_source"]
        arrays["kev"][i] = encode_nested(r.kev)
        arrays["exploits"][i] = encode_nested(r.exploits)
        arrays["cwes"][i] = encode_nested(r.cwes)
        arrays["cpes"][i] = encode_nested(r.cpes)
        cwe_flat = flatten_cwe(r.cwes)
        arrays["primary_cwe_id"][i] = cwe_flat["primary_cwe_id"]
        arrays["primary_cwe_name"][i] = cwe_flat["primary_cwe_name"]
        cpe_flat = flatten_cpe(r.cpes)
        arrays["primary_vendor"][i] = cpe_flat["primary_vendor"]
        arrays["primary_product"][i] = cpe_flat["primary_product"]
        arrays["primary_version"][i] = cpe_flat["primary_version"]
        arrays["references"][i] = encode_nested(r.references)
        arrays["tags"][i] = json.dumps(r.tags)
        arrays["metadata"][i] = json.dumps(r.metadata, default=str)

    dataset._data = arrays
    return dataset


def make_dataset(columns: dict[str, list[Any]]) -> VulnDataset:  # type: ignore[name-defined]
    """Create a VulnDataset from column dict (for testing)."""
    from vulnflow.dataset import VulnDataset

    dataset = VulnDataset()
    dataset._data = {k: np.array(v, dtype=object) for k, v in columns.items()}
    return dataset


class MockExtractor:
    """Test double for Extractor protocol."""

    def __init__(self, name: str = "mock_extractor", num_records: int = 3) -> None:
        self.name = name
        self.num_records = num_records
        self.extract_called = False

    def extract(self) -> VulnDataset:  # type: ignore[name-defined]
        """Return mock VulnDataset."""
        from vulnflow.schema import VulnRecord

        self.extract_called = True
        records = [
            VulnRecord(
                id=uuid4(),
                cve_id=f"CVE-2024-{i:04d}",
                description=f"Mock vulnerability {i} from {self.name}",
            )
            for i in range(self.num_records)
        ]
        return dataset_from_records(records)


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


@pytest.fixture
def nvd_sample_dataset(nvd_sample_path: Path) -> VulnDataset:  # type: ignore[name-defined]
    """Fixture: VulnDataset from NVD sample file."""
    from vulnflow.extractors.nvd import NVDExtractor

    extractor = NVDExtractor(str(nvd_sample_path))
    return extractor.extract()


@pytest.fixture(autouse=True)
def restore_registry():
    """Save and restore registry state for test isolation.

    This fixture runs automatically for all tests. It saves the registry
    state before each test and restores it after, ensuring that tests
    which call clear_registry() don't affect other tests.
    """
    from vulnflow.registry import Registry

    # Save current state
    saved_transforms = Registry._transforms.copy()
    saved_extractors = Registry._extractors.copy()
    saved_loaders = Registry._loaders.copy()
    saved_schemas = Registry._schemas.copy()

    yield

    # Restore state
    Registry._transforms = saved_transforms
    Registry._extractors = saved_extractors
    Registry._loaders = saved_loaders
    Registry._schemas = saved_schemas
