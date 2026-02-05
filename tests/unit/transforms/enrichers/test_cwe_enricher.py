"""Tests for CWEEnricher."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vulnflow.dataset import VulnDataset
from vulnflow.transforms.enrichers.cwe import CWEEnricher


@pytest.mark.asyncio
async def test_cwe_enricher_enriches_existing_cwes(tmp_path: Path) -> None:
    """Test CWEEnricher enriches existing CWE entries with metadata."""
    # Create minimal CWE XML
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

    # Create dataset with partial CWE data (like NVD provides)
    dataset = VulnDataset()
    dataset._data = {
        "cve_id": np.array(["CVE-2024-0001"], dtype=object),
        "cwes": np.array([json.dumps([{"id": "CWE-79"}])], dtype=object),
    }

    # Enrich
    enricher = CWEEnricher(source=cwe_xml, cache_dir=tmp_path / ".cache")
    await enricher.transform(dataset)

    # Verify enrichment
    cwes_raw = dataset.get_column("cwes")[0]
    cwes = json.loads(cwes_raw)

    assert len(cwes) == 1
    assert cwes[0]["id"] == "CWE-79"
    assert cwes[0]["name"] == "Cross-site Scripting (XSS)"
    assert cwes[0]["category_id"] == "CWE-74"
    assert cwes[0]["category_name"] == "Improper Neutralization"
    assert cwes[0]["abstraction_level"] == "Base"


@pytest.mark.asyncio
async def test_cwe_enricher_caches_parsed_database(tmp_path: Path) -> None:
    """Test CWEEnricher caches parsed CWE database."""
    cwe_xml = tmp_path / "cwe.xml"
    cwe_xml.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<Weakness_Catalog>
  <Weaknesses>
    <Weakness ID="79" Name="XSS" Abstraction="Base">
      <Related_Weaknesses />
    </Weakness>
  </Weaknesses>
</Weakness_Catalog>""")

    cache_dir = tmp_path / ".cache"
    enricher = CWEEnricher(source=cwe_xml, cache_dir=cache_dir)

    # First call creates cache
    dataset = VulnDataset()
    dataset._data = {
        "cve_id": np.array(["CVE-2024-0001"], dtype=object),
        "cwes": np.array([json.dumps([{"id": "CWE-79"}])], dtype=object),
    }
    await enricher.transform(dataset)

    # Verify cache exists
    cache_file = cache_dir / "cwe_database.json"
    assert cache_file.exists()

    cached_data = json.loads(cache_file.read_text())
    assert "CWE-79" in cached_data
    assert cached_data["CWE-79"]["name"] == "XSS"
