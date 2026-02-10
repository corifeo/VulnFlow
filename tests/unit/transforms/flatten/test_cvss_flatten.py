"""Tests for CVSSFlattenTransform - CVSS v2 and v3 vector parsing.

Tests:
- v2 vector AV:N/AC:L/Au:N/C:C/I:C/A:C -> cvss_attack_vector == "N"
- v2 vector populates AV, AC, C, I, A; leaves PR, UI, S as None
- v3 vector still works (regression)
- Mixed v2/v3 dataset - both parsed correctly
- Empty/null vector strings -> no crash
"""

from __future__ import annotations

import json

import numpy as np

from vulnflow.dataset import VulnDataset
from vulnflow.transforms.flatten.cvss import CVSSFlattenTransform


def _make_cvss_dataset(entries: list[list[dict]]) -> VulnDataset:
    """Create a VulnDataset with cvss_scores column from entry lists."""
    ds = VulnDataset()
    ds._data = {
        "cve_id": np.array(
            [f"CVE-2024-{i:04d}" for i in range(len(entries))], dtype=object
        ),
        "cvss_scores": np.array(
            [json.dumps(e) for e in entries], dtype=object
        ),
    }
    return ds


class TestCVSSv2Parsing:
    """Tests for CVSS v2 vector string parsing."""

    def test_v2_vector_extracts_attack_vector(self) -> None:
        """v2 vector AV:N -> cvss_attack_vector == 'N'."""
        ds = _make_cvss_dataset([
            [{"version": "2.0", "base_score": 10.0, "base_severity": "HIGH",
              "source": "nvd", "vector_string": "AV:N/AC:L/Au:N/C:C/I:C/A:C",
              "is_authoritative": True}],
        ])
        CVSSFlattenTransform().transform(ds)
        assert ds.get_column("cvss_attack_vector")[0] == "N"

    def test_v2_vector_extracts_attack_complexity(self) -> None:
        """v2 vector AC:L -> cvss_attack_complexity == 'L'."""
        ds = _make_cvss_dataset([
            [{"version": "2.0", "base_score": 10.0, "base_severity": "HIGH",
              "source": "nvd", "vector_string": "AV:N/AC:L/Au:N/C:C/I:C/A:C",
              "is_authoritative": True}],
        ])
        CVSSFlattenTransform().transform(ds)
        assert ds.get_column("cvss_attack_complexity")[0] == "L"

    def test_v2_vector_extracts_impact_metrics(self) -> None:
        """v2 vector populates C, I, A fields."""
        ds = _make_cvss_dataset([
            [{"version": "2.0", "base_score": 10.0, "base_severity": "HIGH",
              "source": "nvd", "vector_string": "AV:N/AC:L/Au:N/C:C/I:C/A:C",
              "is_authoritative": True}],
        ])
        CVSSFlattenTransform().transform(ds)
        assert ds.get_column("cvss_confidentiality")[0] == "C"
        assert ds.get_column("cvss_integrity")[0] == "C"
        assert ds.get_column("cvss_availability")[0] == "C"

    def test_v2_vector_leaves_v3_only_fields_empty(self) -> None:
        """v2 vector leaves PR, UI, S as None."""
        ds = _make_cvss_dataset([
            [{"version": "2.0", "base_score": 10.0, "base_severity": "HIGH",
              "source": "nvd", "vector_string": "AV:N/AC:L/Au:N/C:C/I:C/A:C",
              "is_authoritative": True}],
        ])
        CVSSFlattenTransform().transform(ds)
        assert ds.get_column("cvss_privileges_required")[0] is None
        assert ds.get_column("cvss_user_interaction")[0] is None
        assert ds.get_column("cvss_scope")[0] is None

    def test_v2_local_vector(self) -> None:
        """v2 vector AV:L -> cvss_attack_vector == 'L'."""
        ds = _make_cvss_dataset([
            [{"version": "2.0", "base_score": 4.6, "base_severity": "MEDIUM",
              "source": "nvd", "vector_string": "AV:L/AC:L/Au:N/C:P/I:P/A:P",
              "is_authoritative": True}],
        ])
        CVSSFlattenTransform().transform(ds)
        assert ds.get_column("cvss_attack_vector")[0] == "L"

    def test_v2_adjacent_and_medium(self) -> None:
        """v2 vector with AV:A/AC:M parses correctly."""
        ds = _make_cvss_dataset([
            [{"version": "2.0", "base_score": 5.4, "base_severity": "MEDIUM",
              "source": "nvd", "vector_string": "AV:A/AC:M/Au:S/C:P/I:N/A:P",
              "is_authoritative": True}],
        ])
        CVSSFlattenTransform().transform(ds)
        assert ds.get_column("cvss_attack_vector")[0] == "A"
        assert ds.get_column("cvss_attack_complexity")[0] == "M"
        assert ds.get_column("cvss_confidentiality")[0] == "P"
        assert ds.get_column("cvss_integrity")[0] == "N"
        assert ds.get_column("cvss_availability")[0] == "P"


class TestCVSSv3Regression:
    """Regression tests: v3 parsing still works after v2 addition."""

    def test_v3_vector_still_works(self) -> None:
        """v3.1 vector string still parses all 8 components."""
        ds = _make_cvss_dataset([
            [{"version": "3.1", "base_score": 9.8, "base_severity": "CRITICAL",
              "source": "nvd",
              "vector_string": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
              "is_authoritative": True}],
        ])
        CVSSFlattenTransform().transform(ds)
        assert ds.get_column("cvss_attack_vector")[0] == "N"
        assert ds.get_column("cvss_attack_complexity")[0] == "L"
        assert ds.get_column("cvss_privileges_required")[0] == "N"
        assert ds.get_column("cvss_user_interaction")[0] == "N"
        assert ds.get_column("cvss_scope")[0] == "U"
        assert ds.get_column("cvss_confidentiality")[0] == "H"
        assert ds.get_column("cvss_integrity")[0] == "H"
        assert ds.get_column("cvss_availability")[0] == "H"


class TestCVSSMixedVersions:
    """Tests for datasets with both v2 and v3 records."""

    def test_mixed_v2_v3_dataset(self) -> None:
        """Both v2 and v3 records parsed correctly in same dataset."""
        ds = _make_cvss_dataset([
            # v3 record
            [{"version": "3.1", "base_score": 9.8, "base_severity": "CRITICAL",
              "source": "nvd",
              "vector_string": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
              "is_authoritative": True}],
            # v2 record
            [{"version": "2.0", "base_score": 10.0, "base_severity": "HIGH",
              "source": "nvd", "vector_string": "AV:N/AC:L/Au:N/C:C/I:C/A:C",
              "is_authoritative": True}],
        ])
        CVSSFlattenTransform().transform(ds)

        # v3 record
        assert ds.get_column("cvss_attack_vector")[0] == "N"
        assert ds.get_column("cvss_privileges_required")[0] == "N"
        assert ds.get_column("cvss_confidentiality")[0] == "H"

        # v2 record
        assert ds.get_column("cvss_attack_vector")[1] == "N"
        assert ds.get_column("cvss_privileges_required")[1] is None
        assert ds.get_column("cvss_confidentiality")[1] == "C"


class TestCVSSEdgeCases:
    """Tests for empty/null vector strings."""

    def test_empty_vector_string_no_crash(self) -> None:
        """Empty vector string does not crash."""
        ds = _make_cvss_dataset([
            [{"version": "2.0", "base_score": 5.0, "base_severity": "MEDIUM",
              "source": "nvd", "vector_string": "",
              "is_authoritative": True}],
        ])
        CVSSFlattenTransform().transform(ds)
        assert ds.get_column("cvss_attack_vector")[0] is None

    def test_null_cvss_scores_no_crash(self) -> None:
        """Null cvss_scores entry does not crash."""
        ds = VulnDataset()
        ds._data = {
            "cve_id": np.array(["CVE-2024-0001"], dtype=object),
            "cvss_scores": np.array(["null"], dtype=object),
        }
        CVSSFlattenTransform().transform(ds)
        assert ds.get_column("cvss_attack_vector")[0] is None

    def test_empty_list_cvss_scores_no_crash(self) -> None:
        """Empty list cvss_scores entry does not crash."""
        ds = VulnDataset()
        ds._data = {
            "cve_id": np.array(["CVE-2024-0001"], dtype=object),
            "cvss_scores": np.array(["[]"], dtype=object),
        }
        CVSSFlattenTransform().transform(ds)
        assert ds.get_column("cvss_attack_vector")[0] is None


class TestCVSSv4Flatten:
    """Tests for CVSS v4.0 component field extraction from dict."""

    def test_v4_components_populated(self) -> None:
        """v4.0 entry with component fields populates flatten columns."""
        ds = _make_cvss_dataset([
            [{"version": "4.0", "base_score": 8.5, "base_severity": "HIGH",
              "source": "nvd", "is_authoritative": True,
              "vector_string": "CVSS:4.0/AV:L/AC:L/AT:N/PR:L/UI:N/VC:H/VI:H/VA:H/SC:N/SI:N/SA:N",
              "attack_vector": "LOCAL",
              "attack_complexity": "LOW",
              "attack_requirements": "NONE",
              "privileges_required": "LOW",
              "user_interaction": "NONE",
              "vuln_confidentiality_impact": "HIGH",
              "vuln_integrity_impact": "HIGH",
              "vuln_availability_impact": "HIGH",
              "sub_confidentiality_impact": "NONE",
              "sub_integrity_impact": "NONE",
              "sub_availability_impact": "NONE"}],
        ])
        CVSSFlattenTransform().transform(ds)
        assert ds.get_column("cvss_attack_vector")[0] == "LOCAL"
        assert ds.get_column("cvss_attack_complexity")[0] == "LOW"
        assert ds.get_column("cvss_privileges_required")[0] == "LOW"
        assert ds.get_column("cvss_user_interaction")[0] == "NONE"
        assert ds.get_column("cvss_scope")[0] is None  # v4 has no scope
        # v4 maps vuln_*_impact to the generic columns
        assert ds.get_column("cvss_confidentiality")[0] == "HIGH"
        assert ds.get_column("cvss_integrity")[0] == "HIGH"
        assert ds.get_column("cvss_availability")[0] == "HIGH"

    def test_v31_components_from_dict(self) -> None:
        """v3.1 entry reads components from dict fields, not regex."""
        ds = _make_cvss_dataset([
            [{"version": "3.1", "base_score": 9.8, "base_severity": "CRITICAL",
              "source": "nvd", "is_authoritative": True,
              "vector_string": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
              "attack_vector": "NETWORK",
              "attack_complexity": "LOW",
              "privileges_required": "NONE",
              "user_interaction": "NONE",
              "scope": "UNCHANGED",
              "confidentiality_impact": "HIGH",
              "integrity_impact": "HIGH",
              "availability_impact": "HIGH"}],
        ])
        CVSSFlattenTransform().transform(ds)
        assert ds.get_column("cvss_attack_vector")[0] == "NETWORK"
        assert ds.get_column("cvss_scope")[0] == "UNCHANGED"
        assert ds.get_column("cvss_confidentiality")[0] == "HIGH"
        assert ds.get_column("cvss_integrity")[0] == "HIGH"
        assert ds.get_column("cvss_availability")[0] == "HIGH"

    def test_mixed_v31_v40_all_populated(self) -> None:
        """Dataset with both v3.1 and v4.0 records populates all component columns."""
        ds = _make_cvss_dataset([
            # v3.1 record
            [{"version": "3.1", "base_score": 7.8, "base_severity": "HIGH",
              "source": "nvd", "is_authoritative": True,
              "vector_string": "CVSS:3.1/AV:L/AC:L/PR:L/UI:N/S:U/C:H/I:H/A:H",
              "attack_vector": "LOCAL", "attack_complexity": "LOW",
              "privileges_required": "LOW", "user_interaction": "NONE",
              "scope": "UNCHANGED",
              "confidentiality_impact": "HIGH", "integrity_impact": "HIGH",
              "availability_impact": "HIGH"}],
            # v4.0 record
            [{"version": "4.0", "base_score": 8.5, "base_severity": "HIGH",
              "source": "nvd", "is_authoritative": True,
              "vector_string": "CVSS:4.0/AV:L/AC:L/AT:N/PR:L/UI:N/VC:H/VI:H/VA:H/SC:N/SI:N/SA:N",
              "attack_vector": "LOCAL", "attack_complexity": "LOW",
              "attack_requirements": "NONE", "privileges_required": "LOW",
              "user_interaction": "NONE",
              "vuln_confidentiality_impact": "HIGH", "vuln_integrity_impact": "HIGH",
              "vuln_availability_impact": "HIGH",
              "sub_confidentiality_impact": "NONE", "sub_integrity_impact": "NONE",
              "sub_availability_impact": "NONE"}],
        ])
        CVSSFlattenTransform().transform(ds)

        # v3.1 record - all 8 component columns populated
        assert ds.get_column("cvss_attack_vector")[0] == "LOCAL"
        assert ds.get_column("cvss_scope")[0] == "UNCHANGED"
        assert ds.get_column("cvss_confidentiality")[0] == "HIGH"

        # v4.0 record - all component columns populated (THE BUG FIX)
        assert ds.get_column("cvss_attack_vector")[1] == "LOCAL"
        assert ds.get_column("cvss_scope")[1] is None  # v4 has no scope, expected
        assert ds.get_column("cvss_confidentiality")[1] == "HIGH"  # from vuln_confidentiality_impact
        assert ds.get_column("cvss_integrity")[1] == "HIGH"
        assert ds.get_column("cvss_availability")[1] == "HIGH"
