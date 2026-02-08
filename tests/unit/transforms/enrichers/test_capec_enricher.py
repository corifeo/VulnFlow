"""Tests for CAPECEnricher."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vulnflow.dataset import VulnDataset
from vulnflow.transforms.enrichers.capec import CAPECEnricher

# Minimal CAPEC XML with two attack patterns
CAPEC_XML = """<?xml version="1.0" encoding="UTF-8"?>
<Attack_Pattern_Catalog>
  <Attack_Patterns>
    <Attack_Pattern ID="86" Name="XSS via HTTP Headers" Abstraction="Detailed" Status="Draft">
      <Likelihood_Of_Attack>Medium</Likelihood_Of_Attack>
      <Typical_Severity>High</Typical_Severity>
      <Related_Weaknesses>
        <Related_Weakness CWE_ID="79" />
      </Related_Weaknesses>
    </Attack_Pattern>
    <Attack_Pattern ID="198" Name="XSS Targeting Error Pages" Abstraction="Detailed" Status="Draft">
      <Likelihood_Of_Attack>High</Likelihood_Of_Attack>
      <Typical_Severity>Medium</Typical_Severity>
      <Related_Weaknesses>
        <Related_Weakness CWE_ID="79" />
      </Related_Weaknesses>
    </Attack_Pattern>
    <Attack_Pattern ID="586" Name="Object Injection" Abstraction="Standard" Status="Stable">
      <Likelihood_Of_Attack>Low</Likelihood_Of_Attack>
      <Typical_Severity>Very High</Typical_Severity>
      <Related_Weaknesses>
        <Related_Weakness CWE_ID="502" />
      </Related_Weaknesses>
    </Attack_Pattern>
  </Attack_Patterns>
</Attack_Pattern_Catalog>"""


def _make_dataset_with_cwes(cwe_ids: list[list[str]]) -> VulnDataset:
    """Create dataset with enriched CWE entries."""
    n = len(cwe_ids)
    dataset = VulnDataset()
    cwes_array = np.empty(n, dtype=object)
    cve_ids = np.empty(n, dtype=object)
    for i, ids in enumerate(cwe_ids):
        cve_ids[i] = f"CVE-2024-{i:04d}"
        if ids:
            cwes_array[i] = json.dumps([{"id": cid} for cid in ids])
        else:
            cwes_array[i] = "[]"
    dataset._data = {"cve_id": cve_ids, "cwes": cwes_array}
    return dataset


@pytest.mark.asyncio
async def test_capec_enricher_maps_cwes_to_capecs(tmp_path: Path) -> None:
    """Test CAPECEnricher maps CWE IDs to CAPEC attack patterns with metadata."""
    capec_xml = tmp_path / "capec.xml"
    capec_xml.write_text(CAPEC_XML)

    # Write CWE→CAPEC mapping (from CWEEnricher)
    mitre_dir = tmp_path / ".cache" / "mitre"
    mitre_dir.mkdir(parents=True)
    mapping = {"CWE-79": ["CAPEC-86", "CAPEC-198"]}
    (mitre_dir / "cwe_capec_mapping.json").write_text(json.dumps(mapping))

    dataset = _make_dataset_with_cwes([["CWE-79"]])

    enricher = CAPECEnricher(source=capec_xml, cache_dir=tmp_path / ".cache")
    await enricher.transform(dataset)

    capecs_raw = dataset.get_column("capecs")[0]
    capecs = json.loads(capecs_raw)

    assert len(capecs) == 2
    ids = {c["id"] for c in capecs}
    assert ids == {"CAPEC-86", "CAPEC-198"}

    # Check metadata on CAPEC-86
    capec_86 = next(c for c in capecs if c["id"] == "CAPEC-86")
    assert capec_86["name"] == "XSS via HTTP Headers"
    assert capec_86["abstraction"] == "Detailed"
    assert capec_86["status"] == "Draft"
    assert capec_86["likelihood_of_attack"] == "Medium"
    assert capec_86["typical_severity"] == "High"


@pytest.mark.asyncio
async def test_capec_enricher_caches_parsed_database(tmp_path: Path) -> None:
    """Test CAPECEnricher writes capec_database.json to cache."""
    capec_xml = tmp_path / "capec.xml"
    capec_xml.write_text(CAPEC_XML)

    cache_dir = tmp_path / ".cache"
    mitre_dir = cache_dir / "mitre"
    mitre_dir.mkdir(parents=True)
    (mitre_dir / "cwe_capec_mapping.json").write_text("{}")

    dataset = _make_dataset_with_cwes([["CWE-79"]])
    enricher = CAPECEnricher(source=capec_xml, cache_dir=cache_dir)
    await enricher.transform(dataset)

    cache_file = cache_dir / "mitre" / "capec_database.json"
    assert cache_file.exists()

    cached_data = json.loads(cache_file.read_text())
    assert "CAPEC-86" in cached_data
    assert cached_data["CAPEC-86"]["name"] == "XSS via HTTP Headers"
    assert cached_data["CAPEC-86"]["related_cwes"] == ["CWE-79"]


@pytest.mark.asyncio
async def test_capec_enricher_handles_missing_cwe_mapping(tmp_path: Path) -> None:
    """Test CAPECEnricher works without cwe_capec_mapping.json (CAPEC→CWE only)."""
    capec_xml = tmp_path / "capec.xml"
    capec_xml.write_text(CAPEC_XML)

    # No cwe_capec_mapping.json written — only CAPEC→CWE direction available
    dataset = _make_dataset_with_cwes([["CWE-502"]])

    enricher = CAPECEnricher(source=capec_xml, cache_dir=tmp_path / ".cache")
    await enricher.transform(dataset)

    capecs_raw = dataset.get_column("capecs")[0]
    capecs = json.loads(capecs_raw)

    # Should still find CAPEC-586 via reverse mapping (CAPEC→CWE)
    assert len(capecs) == 1
    assert capecs[0]["id"] == "CAPEC-586"
    assert capecs[0]["name"] == "Object Injection"


@pytest.mark.asyncio
async def test_capec_enricher_deduplicates_from_both_sources(tmp_path: Path) -> None:
    """Test same CAPEC from both mapping directions appears only once."""
    capec_xml = tmp_path / "capec.xml"
    capec_xml.write_text(CAPEC_XML)

    # CWE mapping has CAPEC-86 for CWE-79
    mitre_dir = tmp_path / ".cache" / "mitre"
    mitre_dir.mkdir(parents=True)
    mapping = {"CWE-79": ["CAPEC-86"]}
    (mitre_dir / "cwe_capec_mapping.json").write_text(json.dumps(mapping))

    # CAPEC XML also has CWE-79 → CAPEC-86 (reverse direction)
    # Both sources point to the same CAPEC — should be deduplicated
    dataset = _make_dataset_with_cwes([["CWE-79"]])

    enricher = CAPECEnricher(source=capec_xml, cache_dir=tmp_path / ".cache")
    await enricher.transform(dataset)

    capecs_raw = dataset.get_column("capecs")[0]
    capecs = json.loads(capecs_raw)

    # CAPEC-86 appears from both directions but should only show once
    capec_86_count = sum(1 for c in capecs if c["id"] == "CAPEC-86")
    assert capec_86_count == 1

    # CAPEC-198 appears from reverse direction (CAPEC XML has CWE-79)
    ids = {c["id"] for c in capecs}
    assert "CAPEC-86" in ids
    assert "CAPEC-198" in ids


@pytest.mark.asyncio
async def test_capec_enricher_handles_empty_cwes(tmp_path: Path) -> None:
    """Test records with no CWEs get empty CAPEC array."""
    capec_xml = tmp_path / "capec.xml"
    capec_xml.write_text(CAPEC_XML)

    dataset = _make_dataset_with_cwes([[], ["CWE-79"]])

    mitre_dir = tmp_path / ".cache" / "mitre"
    mitre_dir.mkdir(parents=True)
    (mitre_dir / "cwe_capec_mapping.json").write_text("{}")

    enricher = CAPECEnricher(source=capec_xml, cache_dir=tmp_path / ".cache")
    await enricher.transform(dataset)

    # First record (no CWEs) should have empty capecs
    capecs_0 = json.loads(dataset.get_column("capecs")[0])
    assert capecs_0 == []

    # Second record should have capecs from reverse mapping
    capecs_1 = json.loads(dataset.get_column("capecs")[1])
    assert len(capecs_1) > 0
