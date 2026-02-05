"""Tests for KEVEnricher pattern compliance."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vulnflow.dataset import VulnDataset
from vulnflow.transforms.enrichers.kev import KEVEnricher


@pytest.mark.asyncio
async def test_kev_enricher_creates_json_column_only(tmp_path: Path) -> None:
    """Test KEVEnricher creates kev JSON column (not scalars)."""
    # Create minimal KEV CSV
    kev_csv = tmp_path / "kev.csv"
    kev_csv.write_text(
        "cveID,vendorProject,product,vulnerabilityName,dateAdded,shortDescription,"
        "requiredAction,dueDate,knownRansomwareCampaignUse,notes\n"
        "CVE-2024-0001,Vendor,Product,Test Vuln,2024-01-01,Description,"
        "Apply patch,2024-02-01,Known,Notes"
    )

    # Create dataset
    dataset = VulnDataset()
    dataset.set_column("cve_id", np.array(["CVE-2024-0001"], dtype=object))

    # Enrich
    enricher = KEVEnricher(source=kev_csv, cache_dir=tmp_path / ".cache")
    await enricher.transform(dataset)

    # Verify JSON column exists
    assert "kev" in dataset.columns
    kev_raw = dataset.get_column("kev")[0]
    kev = json.loads(kev_raw)

    assert kev["vendor"] == "Vendor"
    assert kev["product"] == "Product"
    assert kev["vulnerability_name"] == "Test Vuln"
    assert kev["required_action"] == "Apply patch"
    assert kev["known_ransomware_use"] is True

    # Verify scalar columns do NOT exist (will be created by flatten)
    assert "kev_date_added" not in dataset.columns
    assert "kev_vendor" not in dataset.columns


@pytest.mark.asyncio
async def test_kev_enricher_null_for_non_kev(tmp_path: Path) -> None:
    """Test KEVEnricher sets null for CVEs not in KEV."""
    kev_csv = tmp_path / "kev.csv"
    kev_csv.write_text(
        "cveID,vendorProject,product,vulnerabilityName,dateAdded,shortDescription,"
        "requiredAction,dueDate,knownRansomwareCampaignUse,notes\n"
    )

    dataset = VulnDataset()
    dataset.set_column("cve_id", np.array(["CVE-9999-9999"], dtype=object))

    enricher = KEVEnricher(source=kev_csv, cache_dir=tmp_path / ".cache")
    await enricher.transform(dataset)

    kev_raw = dataset.get_column("kev")[0]
    assert kev_raw == "null" or kev_raw is None
