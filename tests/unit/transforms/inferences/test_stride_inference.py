"""Tests for STRIDEInference - Threat category inference using tiered model.

Layer: Tests/Unit
Dependencies: transforms.inferences.stride
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from tests.conftest import make_dataset

from vulnflow.dataset import VulnDataset
from vulnflow.transforms.inferences.stride import STRIDEInference


def _get_threats(ds: VulnDataset, idx: int = 0) -> list[dict[str, Any]]:
    """Parse stride_threats column value at index as list of dicts."""
    raw = ds.get_column("stride_threats")[idx]
    return json.loads(raw)


def _get_categories(ds: VulnDataset, idx: int = 0) -> set[str]:
    """Get set of STRIDE categories assigned to a record."""
    return {t["category"] for t in _get_threats(ds, idx)}


# ============================================================================
# Tier 1: CAPEC consequences
# ============================================================================


class TestTier1CAPECConsequences:
    """Tier 1 maps CAPEC consequence scopes/impacts to STRIDE at 0.95."""

    def test_stride_from_capec_consequences(self, tmp_path: Path) -> None:
        """CAPEC with Confidentiality scope → info_disclosure at 0.95."""
        # Build a CAPEC database with consequences
        capec_db = {
            "CAPEC-86": {
                "name": "XSS via HTTP Headers",
                "consequences": [
                    {"scopes": ["Confidentiality"], "impacts": ["Read Data"]},
                    {"scopes": ["Integrity"], "impacts": ["Modify Data"]},
                ],
            },
        }
        cache_dir = tmp_path / ".cache"
        mitre_dir = cache_dir / "mitre"
        mitre_dir.mkdir(parents=True)
        (mitre_dir / "capec_database.json").write_text(json.dumps(capec_db))

        capecs_json = json.dumps([{"id": "CAPEC-86", "name": "XSS via HTTP Headers"}])
        ds = make_dataset({"cve_id": ["CVE-2024-0001"], "capecs": [capecs_json]})

        STRIDEInference(cache_dir=cache_dir).transform(ds)
        threats = _get_threats(ds)
        categories = {t["category"] for t in threats}

        assert "info_disclosure" in categories
        assert "tampering" in categories

        # Check confidence and source
        tier1 = [t for t in threats if t["source"] == "capec_consequences"]
        assert len(tier1) >= 2
        assert all(t["confidence"] == 0.95 for t in tier1)

        # Check CAPEC ID provenance
        info_threat = next(t for t in tier1 if t["category"] == "info_disclosure")
        assert "CAPEC-86" in info_threat["capec_ids"]


# ============================================================================
# Tier 2: CWE mapping
# ============================================================================


class TestTier2CWEMapping:
    """Tier 2 maps CWE IDs to STRIDE categories at 0.85."""

    def test_stride_from_cwe_mapping(self) -> None:
        """CWE-287 (Improper Authentication) → spoofing at 0.85."""
        ds = make_dataset({"cve_id": ["CVE-2024-0001"], "cwe_id": ["CWE-287"]})
        STRIDEInference().transform(ds)
        threats = _get_threats(ds)

        assert any(
            t["category"] == "spoofing"
            and t["confidence"] == 0.85
            and t["source"] == "cwe_mapping"
            for t in threats
        )

    def test_cwe_multi_category(self) -> None:
        """CWE-79 (XSS) → tampering + info_disclosure."""
        ds = make_dataset({"cve_id": ["CVE-2024-0001"], "cwe_id": ["CWE-79"]})
        STRIDEInference().transform(ds)
        categories = _get_categories(ds)
        assert "tampering" in categories
        assert "info_disclosure" in categories


# ============================================================================
# Tier 3: CVSS impact
# ============================================================================


class TestTier3CVSSImpact:
    """Tier 3 maps CVSS C/I/A high-impact metrics at 0.75."""

    def test_stride_from_cvss_impact(self) -> None:
        """C:H/I:H/A:H → info_disclosure + tampering + dos at 0.75."""
        ds = make_dataset(
            {
                "cve_id": ["CVE-2024-0001"],
                "cvss_confidentiality": ["H"],
                "cvss_integrity": ["H"],
                "cvss_availability": ["H"],
            }
        )
        STRIDEInference().transform(ds)
        threats = _get_threats(ds)
        categories = {t["category"] for t in threats}

        assert "info_disclosure" in categories
        assert "tampering" in categories
        assert "dos" in categories

        cvss_threats = [t for t in threats if t["source"] == "cvss_impact"]
        assert all(t["confidence"] == 0.75 for t in cvss_threats)


# ============================================================================
# Tier 4: Description keywords
# ============================================================================


class TestTier4Description:
    """Tier 4 matches description keywords at 0.60."""

    def test_stride_from_description(self) -> None:
        """'denial of service' keyword → dos at 0.60."""
        ds = make_dataset(
            {
                "cve_id": ["CVE-2024-0001"],
                "description": ["A denial of service vulnerability in the parser"],
            }
        )
        STRIDEInference(confidence_threshold=0.5).transform(ds)
        threats = _get_threats(ds)

        assert any(
            t["category"] == "dos"
            and t["confidence"] == 0.60
            and t["source"] == "description"
            for t in threats
        )


# ============================================================================
# Tier 5: CVSS vector heuristics
# ============================================================================


class TestTier5CVSSVector:
    """Tier 5 uses weak CVSS vector signals at 0.45."""

    def test_stride_from_cvss_vector(self) -> None:
        """PR:N → spoofing at 0.45."""
        ds = make_dataset(
            {
                "cve_id": ["CVE-2024-0001"],
                "cvss_privileges_required": ["N"],
            }
        )
        STRIDEInference(confidence_threshold=0.4).transform(ds)
        threats = _get_threats(ds)

        assert any(
            t["category"] == "spoofing"
            and t["confidence"] == 0.45
            and t["source"] == "cvss_vector"
            for t in threats
        )


# ============================================================================
# Threshold gating
# ============================================================================


class TestThresholdGating:
    """Confidence threshold gates which tiers run."""

    def test_stride_threshold_gates_tiers(self) -> None:
        """threshold=0.90 only runs tier 1 (0.95)."""
        ds = make_dataset(
            {
                "cve_id": ["CVE-2024-0001"],
                "cwe_id": ["CWE-287"],  # Tier 2 would produce spoofing
                "description": ["denial of service"],  # Tier 4 would produce dos
            }
        )
        STRIDEInference(confidence_threshold=0.90).transform(ds)
        threats = _get_threats(ds)

        # Tier 2 (0.85) should NOT run since threshold is 0.90
        assert not any(t["source"] == "cwe_mapping" for t in threats)
        # Tier 4 (0.60) should NOT run
        assert not any(t["source"] == "description" for t in threats)


# ============================================================================
# Tier precedence
# ============================================================================


class TestTierPrecedence:
    """Higher tiers take precedence over lower tiers for same category."""

    def test_stride_higher_tier_wins(self, tmp_path: Path) -> None:
        """Same category from tier 1 and tier 3 → tier 1 confidence kept."""
        capec_db = {
            "CAPEC-86": {
                "name": "XSS",
                "consequences": [
                    {"scopes": ["Confidentiality"], "impacts": ["Read Data"]},
                ],
            },
        }
        cache_dir = tmp_path / ".cache"
        mitre_dir = cache_dir / "mitre"
        mitre_dir.mkdir(parents=True)
        (mitre_dir / "capec_database.json").write_text(json.dumps(capec_db))

        capecs_json = json.dumps([{"id": "CAPEC-86"}])
        ds = make_dataset(
            {
                "cve_id": ["CVE-2024-0001"],
                "capecs": [capecs_json],
                "cvss_confidentiality": ["H"],  # Also produces info_disclosure
            }
        )
        STRIDEInference(cache_dir=cache_dir).transform(ds)
        threats = _get_threats(ds)

        # info_disclosure should come from tier 1 (0.95), not tier 3 (0.75)
        info_threats = [t for t in threats if t["category"] == "info_disclosure"]
        assert len(info_threats) == 1
        assert info_threats[0]["confidence"] == 0.95
        assert info_threats[0]["source"] == "capec_consequences"


# ============================================================================
# Idempotency and overwrite
# ============================================================================


class TestIdempotency:
    """STRIDEInference is idempotent by default, overwrite re-infers."""

    def test_stride_idempotent(self) -> None:
        """Running twice without overwrite produces the same result."""
        ds = make_dataset({"cve_id": ["CVE-2024-0001"], "cwe_id": ["CWE-287"]})
        inference = STRIDEInference()
        inference.transform(ds)
        first_result = ds.get_column("stride_threats")[0]

        inference.transform(ds)
        second_result = ds.get_column("stride_threats")[0]

        assert first_result == second_result

    def test_stride_overwrite(self) -> None:
        """overwrite=True replaces previous inference."""
        ds = make_dataset({"cve_id": ["CVE-2024-0001"], "cwe_id": ["CWE-287"]})
        # First pass: assign spoofing
        STRIDEInference().transform(ds)
        assert "spoofing" in _get_categories(ds)

        # Manually set to something different
        ds.set_column(
            "stride_threats",
            np.array(
                [
                    json.dumps(
                        [
                            {
                                "category": "dos",
                                "confidence": 0.5,
                                "source": "manual",
                                "capec_ids": [],
                            }
                        ]
                    )
                ],
                dtype=object,
            ),
        )

        # Without overwrite, keeps manual value
        STRIDEInference(overwrite=False).transform(ds)
        assert _get_categories(ds) == {"dos"}

        # With overwrite, re-infers from CWE
        STRIDEInference(overwrite=True).transform(ds)
        assert "spoofing" in _get_categories(ds)


# ============================================================================
# Edge cases
# ============================================================================


class TestEdgeCases:
    """Handles missing columns and empty data gracefully."""

    def test_stride_handles_missing_columns(self) -> None:
        """No capecs, no cwes, no cvss → uses lower tiers only."""
        ds = make_dataset(
            {
                "cve_id": ["CVE-2024-0001"],
                "description": ["A denial of service crash vulnerability"],
            }
        )
        STRIDEInference(confidence_threshold=0.5).transform(ds)
        threats = _get_threats(ds)

        # Should still find dos from description
        assert any(t["category"] == "dos" for t in threats)

    def test_no_data_produces_empty_threats(self) -> None:
        """Record with no usable data produces empty threats array."""
        ds = make_dataset({"cve_id": ["CVE-2024-0001"]})
        STRIDEInference().transform(ds)
        threats = _get_threats(ds)
        assert threats == []


# ============================================================================
# Standard pattern alignment
# ============================================================================


class TestStandardPattern:
    """STRIDEInference follows the standard BaseInference pattern."""

    def test_tier_thresholds_is_class_attribute(self) -> None:
        """TIER_THRESHOLDS is defined as a class attribute."""
        assert hasattr(STRIDEInference, "TIER_THRESHOLDS")
        assert isinstance(STRIDEInference.TIER_THRESHOLDS, dict)
        assert 1 in STRIDEInference.TIER_THRESHOLDS
        assert STRIDEInference.TIER_THRESHOLDS[1] == 0.95

    def test_should_run_respects_threshold(self) -> None:
        """STRIDEInference uses BaseInference tier gating."""
        inf = STRIDEInference(confidence_threshold=1.5)
        assert inf._should_run(1) is False

    def test_overwrite_passed_to_base(self) -> None:
        """overwrite param is set via BaseInference.__init__."""
        inf = STRIDEInference(overwrite=True)
        assert inf.overwrite is True

    def test_tiers_passed_to_base(self) -> None:
        """tiers param is set via BaseInference.__init__."""
        inf = STRIDEInference(tiers=[1, 2])
        assert inf.tiers == [1, 2]

    def test_explicit_tier_selection(self) -> None:
        """Explicit tiers=[2] only runs tier 2."""
        ds = make_dataset(
            {
                "cve_id": ["CVE-2024-0001"],
                "cwe_id": ["CWE-287"],
                "description": ["denial of service"],
            }
        )
        STRIDEInference(tiers=[2]).transform(ds)
        threats = _get_threats(ds)

        # Tier 2 (CWE mapping) should run
        assert any(t["source"] == "cwe_mapping" for t in threats)
        # Tier 4 (description) should NOT run
        assert not any(t["source"] == "description" for t in threats)

    def test_array_init_uses_np_full(self) -> None:
        """Empty stride_threats values are None, not uninitialized."""
        ds = make_dataset({"cve_id": ["CVE-2024-0001", "CVE-2024-0002"]})
        # Only run tier 1, which needs CAPEC data we don't have
        STRIDEInference(tiers=[1]).transform(ds)
        # Records with no data should get "[]", not garbage from np.empty
        for i in range(2):
            raw = ds.get_column("stride_threats")[i]
            assert raw is not None
            parsed = json.loads(raw)
            assert isinstance(parsed, list)
