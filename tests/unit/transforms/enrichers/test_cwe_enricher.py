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

    # Verify cache exists in mitre/ subdirectory
    cache_file = cache_dir / "mitre" / "cwe_database.json"
    assert cache_file.exists()

    cached_data = json.loads(cache_file.read_text())
    assert "CWE-79" in cached_data
    assert cached_data["CWE-79"]["name"] == "XSS"


@pytest.mark.asyncio
async def test_cwe_enricher_writes_capec_mapping(tmp_path: Path) -> None:
    """Test CWEEnricher writes cwe_capec_mapping.json to mitre/ cache."""
    cwe_xml = tmp_path / "cwe.xml"
    cwe_xml.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<Weakness_Catalog>
  <Weaknesses>
    <Weakness ID="79" Name="XSS" Abstraction="Base">
      <Related_Weaknesses />
      <Related_Attack_Patterns>
        <Related_Attack_Pattern CAPEC_ID="86" />
        <Related_Attack_Pattern CAPEC_ID="198" />
      </Related_Attack_Patterns>
    </Weakness>
    <Weakness ID="502" Name="Deserialization" Abstraction="Base">
      <Related_Weaknesses />
      <Related_Attack_Patterns>
        <Related_Attack_Pattern CAPEC_ID="586" />
      </Related_Attack_Patterns>
    </Weakness>
  </Weaknesses>
</Weakness_Catalog>""")

    cache_dir = tmp_path / ".cache"
    enricher = CWEEnricher(source=cwe_xml, cache_dir=cache_dir)

    dataset = VulnDataset()
    dataset._data = {
        "cve_id": np.array(["CVE-2024-0001"], dtype=object),
        "cwes": np.array([json.dumps([{"id": "CWE-79"}])], dtype=object),
    }
    await enricher.transform(dataset)

    mapping_file = cache_dir / "mitre" / "cwe_capec_mapping.json"
    assert mapping_file.exists()

    mapping = json.loads(mapping_file.read_text())
    assert "CWE-79" in mapping
    assert "CAPEC-86" in mapping["CWE-79"]
    assert "CAPEC-198" in mapping["CWE-79"]
    assert "CWE-502" in mapping
    assert mapping["CWE-502"] == ["CAPEC-586"]


@pytest.mark.asyncio
async def test_cwe_enricher_uses_mitre_cache_dir(tmp_path: Path) -> None:
    """Test CWEEnricher stores cache files in mitre/ not cwe/."""
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

    dataset = VulnDataset()
    dataset._data = {
        "cve_id": np.array(["CVE-2024-0001"], dtype=object),
        "cwes": np.array([json.dumps([{"id": "CWE-79"}])], dtype=object),
    }
    await enricher.transform(dataset)

    # mitre/ directory exists with cache files
    assert (cache_dir / "mitre" / "cwe_database.json").exists()
    assert (cache_dir / "mitre" / "cwe_capec_mapping.json").exists()
    # cwe/ directory should NOT exist
    assert not (cache_dir / "cwe").exists()
