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
