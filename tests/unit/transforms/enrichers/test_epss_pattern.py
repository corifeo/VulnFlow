"""Tests for EPSSEnricher pattern compliance."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vulnflow.dataset import VulnDataset
from vulnflow.transforms.enrichers.epss import EPSSEnricher


@pytest.mark.asyncio
async def test_epss_enricher_creates_array_column_only(tmp_path: Path) -> None:
    """Test EPSSEnricher creates epss_scores JSON array (not scalars)."""
    # Create minimal EPSS CSV
    epss_csv = tmp_path / "epss.csv"
    epss_csv.write_text(
        "#model_version:v2023.03.01\n"
        "#score_date:2024-01-01\n"
        "cve,epss,percentile\n"
        "CVE-2024-0001,0.5,0.85\n"
    )

    # Create dataset
    dataset = VulnDataset()
    dataset.set_column("cve_id", np.array(["CVE-2024-0001"], dtype=object))
    dataset.set_column("epss_scores", np.array(["[]"], dtype=object))

    # Enrich
    enricher = EPSSEnricher(source=epss_csv, cache_dir=tmp_path / ".cache")
    await enricher.transform(dataset)

    # Verify JSON array column exists
    assert "epss_scores" in dataset.columns
    scores_raw = dataset.get_column("epss_scores")[0]
    scores = json.loads(scores_raw)

    assert len(scores) == 1
    assert scores[0]["probability"] == 0.5
    assert scores[0]["percentile"] == 0.85

    # Verify scalar columns do NOT exist (will be created by flatten)
    assert "epss_probability" not in dataset.columns
    assert "epss_percentile" not in dataset.columns


@pytest.mark.asyncio
async def test_epss_enricher_preserves_unmatched_records(tmp_path: Path) -> None:
    """Non-matching records should not be json-round-tripped."""
    epss_csv = tmp_path / "epss.csv"
    epss_csv.write_text(
        "#model_version:v2023.03.01,score_date:2024-01-01\n"
        "cve,epss,percentile\n"
        "CVE-2024-0001,0.5,0.85\n"
    )

    dataset = VulnDataset()
    dataset.set_column(
        "cve_id",
        np.array(["CVE-2024-0001", "CVE-9999-0001"], dtype=object),
    )
    dataset.set_column(
        "epss_scores",
        np.array(["[]", "[]"], dtype=object),
    )

    enricher = EPSSEnricher(source=epss_csv, cache_dir=tmp_path / ".cache")
    await enricher.transform(dataset)

    # Matched record should have EPSS data
    scores_0 = json.loads(dataset.get_column("epss_scores")[0])
    assert len(scores_0) == 1
    assert scores_0[0]["probability"] == 0.5

    # Unmatched record should be unchanged
    assert dataset.get_column("epss_scores")[1] == "[]"
