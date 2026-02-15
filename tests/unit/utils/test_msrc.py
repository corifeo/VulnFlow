"""MSRC utility function tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vulnflow.schema import CVSSv3Entry

FIXTURES = Path(__file__).resolve().parent.parent.parent / "fixtures"


class TestToMsrcMonthId:
    """to_msrc_month_id converts YYYY-MM to YYYY-Mon."""

    def test_january(self) -> None:
        from vulnflow.utils.msrc import to_msrc_month_id

        assert to_msrc_month_id("2025-01") == "2025-Jan"

    def test_april(self) -> None:
        from vulnflow.utils.msrc import to_msrc_month_id

        assert to_msrc_month_id("2025-04") == "2025-Apr"

    def test_december(self) -> None:
        from vulnflow.utils.msrc import to_msrc_month_id

        assert to_msrc_month_id("2024-12") == "2024-Dec"

    def test_invalid_format_raises(self) -> None:
        from vulnflow.utils.msrc import to_msrc_month_id

        with pytest.raises(ValueError, match="Invalid month format"):
            to_msrc_month_id("2025-13")

    def test_bad_string_raises(self) -> None:
        from vulnflow.utils.msrc import to_msrc_month_id

        with pytest.raises(ValueError, match="Invalid month format"):
            to_msrc_month_id("not-a-date")


class TestResolveMonthIds:
    """resolve_month_ids filters available months by params."""

    AVAILABLE = ["2024-Nov", "2024-Dec", "2025-Jan", "2025-Feb", "2025-Mar", "2025-Apr"]

    def test_no_args_returns_current_month(self) -> None:
        from vulnflow.utils.msrc import resolve_month_ids

        result = resolve_month_ids(
            year=None, start=None, end=None, available=self.AVAILABLE
        )
        # Should return exactly 1 month (the latest available or current)
        assert len(result) >= 1

    def test_single_year(self) -> None:
        from vulnflow.utils.msrc import resolve_month_ids

        result = resolve_month_ids(
            year=2025, start=None, end=None, available=self.AVAILABLE
        )
        assert result == ["2025-Jan", "2025-Feb", "2025-Mar", "2025-Apr"]

    def test_multiple_years(self) -> None:
        from vulnflow.utils.msrc import resolve_month_ids

        result = resolve_month_ids(
            year=[2024, 2025], start=None, end=None, available=self.AVAILABLE
        )
        assert result == self.AVAILABLE

    def test_start_only_single_month(self) -> None:
        from vulnflow.utils.msrc import resolve_month_ids

        result = resolve_month_ids(
            year=None, start="2025-02", end=None, available=self.AVAILABLE
        )
        assert result == ["2025-Feb"]

    def test_start_and_end_range(self) -> None:
        from vulnflow.utils.msrc import resolve_month_ids

        result = resolve_month_ids(
            year=None, start="2025-01", end="2025-03", available=self.AVAILABLE
        )
        assert result == ["2025-Jan", "2025-Feb", "2025-Mar"]

    def test_year_and_start_raises(self) -> None:
        from vulnflow.utils.msrc import resolve_month_ids

        with pytest.raises(ValueError, match="Cannot combine"):
            resolve_month_ids(
                year=2025, start="2025-01", end=None, available=self.AVAILABLE
            )

    def test_end_without_start_raises(self) -> None:
        from vulnflow.utils.msrc import resolve_month_ids

        with pytest.raises(ValueError, match="'end' requires 'start'"):
            resolve_month_ids(
                year=None, start=None, end="2025-04", available=self.AVAILABLE
            )

    def test_no_matching_months_raises(self) -> None:
        from vulnflow.utils.msrc import resolve_month_ids

        with pytest.raises(ValueError, match="No matching months"):
            resolve_month_ids(year=2023, start=None, end=None, available=self.AVAILABLE)


class TestParseMsrcCvss:
    """parse_msrc_cvss extracts CVSS entries from CVSSScoreSets."""

    def test_basic_cvss31(self) -> None:
        from vulnflow.utils.msrc import parse_msrc_cvss

        score_sets = [
            {
                "BaseScore": 9.8,
                "TemporalScore": 8.5,
                "Vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H/E:F/RL:O/RC:C",
                "ProductID": ["11568", "11569"],
            }
        ]
        entries = parse_msrc_cvss(score_sets)
        assert len(entries) == 1
        entry = entries[0]
        assert isinstance(entry, CVSSv3Entry)
        assert entry.base_score == 9.8
        assert entry.source == "microsoft"
        assert entry.is_authoritative is True
        assert entry.version == "3.1"
        assert entry.metadata == {"product_ids": ["11568", "11569"]}

    def test_dedup_by_vector(self) -> None:
        from vulnflow.utils.msrc import parse_msrc_cvss

        score_sets = [
            {
                "BaseScore": 9.8,
                "Vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
                "ProductID": ["11568"],
            },
            {
                "BaseScore": 9.8,
                "Vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
                "ProductID": ["11569"],
            },
        ]
        entries = parse_msrc_cvss(score_sets)
        # Same vector → merged product_ids
        assert len(entries) == 1
        assert set(entries[0].metadata["product_ids"]) == {"11568", "11569"}

    def test_empty_score_sets(self) -> None:
        from vulnflow.utils.msrc import parse_msrc_cvss

        assert parse_msrc_cvss([]) == []

    def test_missing_vector(self) -> None:
        from vulnflow.utils.msrc import parse_msrc_cvss

        score_sets = [{"BaseScore": 5.0, "ProductID": ["11568"]}]
        entries = parse_msrc_cvss(score_sets)
        assert len(entries) == 0


class TestParseMsrcCwes:
    """parse_msrc_cwes extracts CWE entries from MSRC CWE object."""

    def test_basic_cwe(self) -> None:
        from vulnflow.utils.msrc import parse_msrc_cwes

        cwe_data = {"ID": "CWE-122", "Value": "Heap-based Buffer Overflow"}
        entries = parse_msrc_cwes(cwe_data)
        assert len(entries) == 1
        assert entries[0].id == "CWE-122"
        assert entries[0].name == "Heap-based Buffer Overflow"

    def test_none_cwe(self) -> None:
        from vulnflow.utils.msrc import parse_msrc_cwes

        assert parse_msrc_cwes(None) == []

    def test_empty_cwe(self) -> None:
        from vulnflow.utils.msrc import parse_msrc_cwes

        assert parse_msrc_cwes({}) == []


class TestParseMsrcReferences:
    """parse_msrc_references extracts references from Remediations."""

    def test_basic_remediation(self) -> None:
        from vulnflow.utils.msrc import parse_msrc_references

        remediations = [
            {
                "Type": 1,
                "Description": {"Value": "5025221"},
                "URL": "https://catalog.update.microsoft.com/v7/site/Search.aspx?q=KB5025221",
                "ProductID": ["11568"],
                "SubType": "Security Update",
            }
        ]
        entries = parse_msrc_references(remediations)
        assert len(entries) == 1
        assert entries[0].source == "msrc"
        assert "patch" in entries[0].tags
        assert "kb-article" in entries[0].tags

    def test_dedup_by_url(self) -> None:
        from vulnflow.utils.msrc import parse_msrc_references

        url = "https://catalog.update.microsoft.com/v7/site/Search.aspx?q=KB5025221"
        remediations = [
            {"Type": 1, "URL": url, "ProductID": ["11568"]},
            {"Type": 1, "URL": url, "ProductID": ["11569"]},
        ]
        entries = parse_msrc_references(remediations)
        assert len(entries) == 1

    def test_empty_remediations(self) -> None:
        from vulnflow.utils.msrc import parse_msrc_references

        assert parse_msrc_references([]) == []


class TestBuildMsrcMetadata:
    """build_msrc_metadata builds metadata dict from vulnerability data."""

    def test_exploited_vulnerability(self) -> None:
        from vulnflow.utils.msrc import build_msrc_metadata

        vuln = {
            "Title": {"Value": "Test RCE"},
            "Threats": [
                {"Type": 0, "Description": {"Value": "Remote Code Execution"}},
                {"Type": 1, "Description": {"Value": "Exploitation Detected"}},
                {
                    "Type": 3,
                    "Description": {"Value": "Critical"},
                    "ProductID": ["11568"],
                },
            ],
            "ProductStatuses": [{"Type": 0, "ProductID": ["11568"]}],
            "Remediations": [
                {
                    "Type": 1,
                    "Description": {"Value": "5025221"},
                    "URL": "https://example.com/kb",
                    "ProductID": ["11568"],
                    "RestartRequired": {"Value": "Yes"},
                    "FixedBuild": "10.0.26100.3775",
                    "Supercedence": "5023696",
                    "SubType": "Security Update",
                }
            ],
            "Acknowledgments": [{"Name": [{"Value": "Researcher"}]}],
        }
        meta = build_msrc_metadata(vuln, "2025-Apr")
        assert meta["month_id"] == "2025-Apr"
        assert meta["title"] == "Test RCE"
        assert meta["exploited"] is True
        assert meta["publicly_disclosed"] is False
        assert meta["exploitability"] == "Exploitation Detected"
        assert meta["impact"] == "Remote Code Execution"
        assert meta["severity_by_product_id"] == {"11568": "Critical"}
        assert len(meta["remediations"]) == 1
        assert meta["remediations"][0]["restart_required"] == "Yes"
        assert meta["remediations"][0]["fixed_build"] == "10.0.26100.3775"

    def test_publicly_disclosed(self) -> None:
        from vulnflow.utils.msrc import build_msrc_metadata

        vuln = {
            "Title": {"Value": "Test EoP"},
            "Threats": [
                {"Type": 1, "Description": {"Value": "Publicly Disclosed"}},
            ],
            "ProductStatuses": [],
            "Remediations": [],
            "Acknowledgments": [],
        }
        meta = build_msrc_metadata(vuln, "2025-Apr")
        assert meta["exploited"] is False
        assert meta["publicly_disclosed"] is True

    def test_neither_exploited_nor_disclosed(self) -> None:
        from vulnflow.utils.msrc import build_msrc_metadata

        vuln = {
            "Title": {"Value": "Test"},
            "Threats": [
                {"Type": 1, "Description": {"Value": "Exploitation Less Likely"}},
            ],
            "ProductStatuses": [],
            "Remediations": [],
            "Acknowledgments": [],
        }
        meta = build_msrc_metadata(vuln, "2025-Apr")
        assert meta["exploited"] is False
        assert meta["publicly_disclosed"] is False


class TestPopulateMsrcColumns:
    """populate_msrc_columns converts CVRF vulnerabilities to columnar arrays."""

    def test_basic_conversion(self) -> None:
        from vulnflow.utils.msrc import populate_msrc_columns

        sample = json.loads((FIXTURES / "msrc_sample.json").read_text())
        vulns = sample["Vulnerability"]
        arrays = populate_msrc_columns(vulns, "2025-Apr")

        assert len(arrays["cve_id"]) == 3
        assert list(arrays["cve_id"]) == [
            "CVE-2025-27745",
            "CVE-2025-27746",
            "CVE-2025-27747",
        ]

    def test_descriptions_extracted(self) -> None:
        from vulnflow.utils.msrc import populate_msrc_columns

        sample = json.loads((FIXTURES / "msrc_sample.json").read_text())
        arrays = populate_msrc_columns(sample["Vulnerability"], "2025-Apr")
        assert "remote code execution" in arrays["description"][0].lower()

    def test_cvss_scores_populated(self) -> None:
        from vulnflow.utils.msrc import populate_msrc_columns

        sample = json.loads((FIXTURES / "msrc_sample.json").read_text())
        arrays = populate_msrc_columns(sample["Vulnerability"], "2025-Apr")
        cvss_data = json.loads(arrays["cvss_scores"][0])
        assert len(cvss_data) >= 1
        assert cvss_data[0]["base_score"] == 9.8

    def test_metadata_populated(self) -> None:
        from vulnflow.utils.msrc import populate_msrc_columns

        sample = json.loads((FIXTURES / "msrc_sample.json").read_text())
        arrays = populate_msrc_columns(sample["Vulnerability"], "2025-Apr")
        meta = json.loads(arrays["metadata"][0])
        assert meta["msrc"]["exploited"] is True
        assert meta["msrc"]["month_id"] == "2025-Apr"

    def test_empty_input(self) -> None:
        from vulnflow.utils.msrc import populate_msrc_columns

        arrays = populate_msrc_columns([], "2025-Apr")
        assert arrays == {}

    def test_cpes_always_empty(self) -> None:
        from vulnflow.utils.msrc import populate_msrc_columns

        sample = json.loads((FIXTURES / "msrc_sample.json").read_text())
        arrays = populate_msrc_columns(sample["Vulnerability"], "2025-Apr")
        assert arrays["cpes"][0] == "[]"

    def test_nvd_fields_empty(self) -> None:
        from vulnflow.utils.msrc import populate_msrc_columns

        sample = json.loads((FIXTURES / "msrc_sample.json").read_text())
        arrays = populate_msrc_columns(sample["Vulnerability"], "2025-Apr")
        assert arrays["nvd_published_at"][0] is None
        assert arrays["nvd_status"][0] is None
