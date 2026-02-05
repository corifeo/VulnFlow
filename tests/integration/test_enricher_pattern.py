"""Integration tests for enricher → flattener pattern."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vulnflow.dataset import VulnDataset
from vulnflow.transforms.enrichers.cwe import CWEEnricher
from vulnflow.transforms.enrichers.kev import KEVEnricher
from vulnflow.transforms.flatten.cwe import CWEFlattenTransform
from vulnflow.transforms.flatten.kev import KEVFlattenTransform


@pytest.mark.asyncio
async def test_kev_enricher_to_flattener_pipeline(tmp_path: Path) -> None:
    """Test KEV enricher → flattener produces correct scalars."""
    # Setup KEV data
    kev_csv = tmp_path / "kev.csv"
    kev_csv.write_text(
        "cveID,vendorProject,product,vulnerabilityName,dateAdded,shortDescription,"
        "requiredAction,dueDate,knownRansomwareCampaignUse,notes\n"
        "CVE-2024-0001,Apache,Log4j,RCE,2024-01-01,Critical RCE,"
        "Apply patch,2024-02-01,Known,Update now\n"
    )

    # Create dataset
    dataset = VulnDataset()
    dataset.set_column("cve_id", np.array(["CVE-2024-0001", "CVE-9999-9999"], dtype=object))

    # Run enricher
    enricher = KEVEnricher(source=kev_csv, cache_dir=tmp_path / ".cache")
    await enricher.transform(dataset)

    # Verify JSON column exists
    assert "kev" in dataset.columns
    kev_raw = dataset.get_column("kev")[0]
    kev = json.loads(kev_raw)
    assert kev["vendor"] == "Apache"

    # Run flattener
    flattener = KEVFlattenTransform()
    flattener.transform(dataset)

    # Verify scalars exist and are correct
    assert dataset.get_column("is_kev")[0] is True
    assert dataset.get_column("is_kev")[1] is False
    assert dataset.get_column("kev_vendor")[0] == "Apache"
    assert dataset.get_column("kev_product")[0] == "Log4j"


@pytest.mark.asyncio
async def test_cwe_enricher_to_flattener_pipeline(tmp_path: Path) -> None:
    """Test CWE enricher → flattener produces complete metadata."""
    # Setup CWE data
    cwe_xml = tmp_path / "cwe.xml"
    cwe_xml.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<Weakness_Catalog>
  <Weaknesses>
    <Weakness ID="79" Name="Cross-site Scripting (XSS)" Abstraction="Base">
      <Related_Weaknesses>
        <Related_Weakness Nature="ChildOf" CWE_ID="74" />
      </Related_Weaknesses>
    </Weakness>
    <Weakness ID="74" Name="Improper Neutralization" Abstraction="Class">
      <Related_Weaknesses />
    </Weakness>
  </Weaknesses>
</Weakness_Catalog>""")

    # Create dataset with partial CWE (like NVD provides)
    dataset = VulnDataset()
    dataset.set_column("cve_id", np.array(["CVE-2024-0001"], dtype=object))
    dataset.set_column("cwes", np.array([json.dumps([{"id": "CWE-79"}])], dtype=object))

    # Run enricher
    enricher = CWEEnricher(source=cwe_xml, cache_dir=tmp_path / ".cache")
    await enricher.transform(dataset)

    # Verify enrichment
    cwes_raw = dataset.get_column("cwes")[0]
    cwes = json.loads(cwes_raw)
    assert cwes[0]["name"] == "Cross-site Scripting (XSS)"
    assert cwes[0]["category_name"] == "Improper Neutralization"

    # Run flattener
    flattener = CWEFlattenTransform()
    flattener.transform(dataset)

    # Verify all scalar fields populated
    assert dataset.get_column("cwe_id")[0] == "CWE-79"
    assert dataset.get_column("cwe_name")[0] == "Cross-site Scripting (XSS)"
    assert dataset.get_column("cwe_category_id")[0] == "CWE-74"
    assert dataset.get_column("cwe_category_name")[0] == "Improper Neutralization"
    assert dataset.get_column("cwe_abstraction_level")[0] == "Base"
