"""Tests for schema dataclass to_dict() methods."""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime

from vulnflow.schema import (
    CPEEntry,
    CVSSv3Entry,
    CVSSv4Entry,
    CWEEntry,
    ReferenceEntry,
)


class TestCVSSv3EntryToDict:
    def test_to_dict_matches_asdict_simple(self) -> None:
        entry = CVSSv3Entry(
            version="3.1",
            vector_string="CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
            base_score=9.8,
            base_severity="CRITICAL",
            source="nvd",
        )
        result = entry.to_dict()
        expected = asdict(entry)
        assert result == expected

    def test_to_dict_with_datetime_fields(self) -> None:
        dt = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)
        entry = CVSSv3Entry(
            version="3.1",
            vector_string="CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
            base_score=9.8,
            base_severity="CRITICAL",
            source="nvd",
            published_at=dt,
        )
        result = entry.to_dict()
        assert result["published_at"] == dt
        assert result["updated_at"] is None

    def test_to_dict_includes_all_v3_fields(self) -> None:
        entry = CVSSv3Entry(
            version="3.1",
            vector_string="AV:N",
            base_score=5.0,
            base_severity="MEDIUM",
            source="nvd",
        )
        result = entry.to_dict()
        assert set(result.keys()) == {
            "version",
            "vector_string",
            "base_score",
            "base_severity",
            "source",
            "is_authoritative",
            "attack_vector",
            "attack_complexity",
            "privileges_required",
            "user_interaction",
            "scope",
            "confidentiality_impact",
            "integrity_impact",
            "availability_impact",
            "exploitability_score",
            "impact_score",
            "source_url",
            "published_at",
            "updated_at",
        }

    def test_to_dict_does_not_contain_v4_fields(self) -> None:
        entry = CVSSv3Entry(
            version="3.1",
            vector_string="AV:N",
            base_score=5.0,
            base_severity="MEDIUM",
            source="nvd",
        )
        result = entry.to_dict()
        assert "vuln_confidentiality_impact" not in result
        assert "attack_requirements" not in result
        assert "sub_confidentiality_impact" not in result
        assert "exploit_maturity" not in result

    def test_v3_metric_fields(self) -> None:
        entry = CVSSv3Entry(
            version="3.1",
            vector_string="CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H",
            base_score=10.0,
            base_severity="CRITICAL",
            source="nvd",
            attack_vector="NETWORK",
            attack_complexity="LOW",
            privileges_required="NONE",
            user_interaction="NONE",
            scope="CHANGED",
            confidentiality_impact="HIGH",
            integrity_impact="HIGH",
            availability_impact="HIGH",
            exploitability_score=3.9,
            impact_score=6.0,
        )
        result = entry.to_dict()
        assert result["attack_vector"] == "NETWORK"
        assert result["attack_complexity"] == "LOW"
        assert result["privileges_required"] == "NONE"
        assert result["user_interaction"] == "NONE"
        assert result["scope"] == "CHANGED"
        assert result["confidentiality_impact"] == "HIGH"
        assert result["integrity_impact"] == "HIGH"
        assert result["availability_impact"] == "HIGH"
        assert result["exploitability_score"] == 3.9
        assert result["impact_score"] == 6.0


class TestCVSSv4EntryToDict:
    def test_to_dict_matches_asdict_simple(self) -> None:
        entry = CVSSv4Entry(
            version="4.0",
            vector_string="CVSS:4.0/AV:N/AC:L/AT:N/PR:N/UI:N/VC:H/VI:H/VA:H/SC:N/SI:N/SA:N",
            base_score=9.3,
            base_severity="CRITICAL",
            source="nvd",
        )
        result = entry.to_dict()
        expected = asdict(entry)
        assert result == expected

    def test_to_dict_includes_all_v4_fields(self) -> None:
        entry = CVSSv4Entry(
            version="4.0",
            vector_string="AV:N",
            base_score=5.0,
            base_severity="MEDIUM",
            source="nvd",
        )
        result = entry.to_dict()
        assert set(result.keys()) == {
            "version",
            "vector_string",
            "base_score",
            "base_severity",
            "source",
            "is_authoritative",
            "attack_vector",
            "attack_complexity",
            "attack_requirements",
            "privileges_required",
            "user_interaction",
            "vuln_confidentiality_impact",
            "vuln_integrity_impact",
            "vuln_availability_impact",
            "sub_confidentiality_impact",
            "sub_integrity_impact",
            "sub_availability_impact",
            "exploit_maturity",
            "automatable",
            "recovery",
            "safety",
            "provider_urgency",
            "vulnerability_response_effort",
            "value_density",
            "exploitability_score",
            "impact_score",
            "source_url",
            "published_at",
            "updated_at",
        }

    def test_to_dict_does_not_contain_v3_fields(self) -> None:
        entry = CVSSv4Entry(
            version="4.0",
            vector_string="AV:N",
            base_score=5.0,
            base_severity="MEDIUM",
            source="nvd",
        )
        result = entry.to_dict()
        assert "scope" not in result
        assert "confidentiality_impact" not in result
        assert "integrity_impact" not in result
        assert "availability_impact" not in result

    def test_v4_base_metric_fields(self) -> None:
        entry = CVSSv4Entry(
            version="4.0",
            vector_string="CVSS:4.0/AV:N/AC:L/AT:N/PR:N/UI:N/VC:H/VI:H/VA:H/SC:N/SI:N/SA:N",
            base_score=9.3,
            base_severity="CRITICAL",
            source="nvd",
            attack_vector="NETWORK",
            attack_complexity="LOW",
            attack_requirements="NONE",
            privileges_required="NONE",
            user_interaction="NONE",
            vuln_confidentiality_impact="HIGH",
            vuln_integrity_impact="HIGH",
            vuln_availability_impact="HIGH",
            sub_confidentiality_impact="NONE",
            sub_integrity_impact="NONE",
            sub_availability_impact="NONE",
        )
        result = entry.to_dict()
        assert result["attack_vector"] == "NETWORK"
        assert result["attack_complexity"] == "LOW"
        assert result["attack_requirements"] == "NONE"
        assert result["privileges_required"] == "NONE"
        assert result["user_interaction"] == "NONE"
        assert result["vuln_confidentiality_impact"] == "HIGH"
        assert result["vuln_integrity_impact"] == "HIGH"
        assert result["vuln_availability_impact"] == "HIGH"
        assert result["sub_confidentiality_impact"] == "NONE"
        assert result["sub_integrity_impact"] == "NONE"
        assert result["sub_availability_impact"] == "NONE"

    def test_v4_supplemental_metrics(self) -> None:
        entry = CVSSv4Entry(
            version="4.0",
            vector_string="AV:N",
            base_score=8.0,
            base_severity="HIGH",
            source="nvd",
            exploit_maturity="ATTACKED",
            automatable="YES",
            recovery="AUTOMATIC",
            safety="NEGLIGIBLE",
            provider_urgency="RED",
            vulnerability_response_effort="LOW",
            value_density="CONCENTRATED",
        )
        result = entry.to_dict()
        assert result["exploit_maturity"] == "ATTACKED"
        assert result["automatable"] == "YES"
        assert result["recovery"] == "AUTOMATIC"
        assert result["safety"] == "NEGLIGIBLE"
        assert result["provider_urgency"] == "RED"
        assert result["vulnerability_response_effort"] == "LOW"
        assert result["value_density"] == "CONCENTRATED"

    def test_v4_supplemental_metrics_default_none(self) -> None:
        entry = CVSSv4Entry(
            version="4.0",
            vector_string="AV:N",
            base_score=5.0,
            base_severity="MEDIUM",
            source="nvd",
        )
        result = entry.to_dict()
        assert result["exploit_maturity"] is None
        assert result["automatable"] is None
        assert result["recovery"] is None
        assert result["safety"] is None
        assert result["provider_urgency"] is None
        assert result["vulnerability_response_effort"] is None
        assert result["value_density"] is None

    def test_to_dict_with_datetime_fields(self) -> None:
        dt = datetime(2024, 6, 1, 10, 30, 0, tzinfo=UTC)
        entry = CVSSv4Entry(
            version="4.0",
            vector_string="AV:N",
            base_score=7.0,
            base_severity="HIGH",
            source="nvd",
            published_at=dt,
            updated_at=dt,
        )
        result = entry.to_dict()
        assert result["published_at"] == dt
        assert result["updated_at"] == dt


class TestCWEEntryToDict:
    def test_to_dict_matches_asdict(self) -> None:
        entry = CWEEntry(
            id="CWE-502",
            name="Deserialization of Untrusted Data",
            category_id="CWE-913",
            abstraction_level="Base",
            consequences=[{"scopes": ["Integrity"], "impacts": ["Execute Unauthorized Code"]}],
            applicable_platforms=["Java", "Python"],
            modes_of_introduction=["Implementation"],
        )
        assert entry.to_dict() == asdict(entry)

    def test_to_dict_with_none_fields(self) -> None:
        entry = CWEEntry(id="CWE-79")
        result = entry.to_dict()
        assert result["id"] == "CWE-79"
        assert result["name"] is None
        assert result["consequences"] is None


class TestCPEEntryToDict:
    def test_to_dict_matches_asdict(self) -> None:
        entry = CPEEntry(vendor="apache", product="log4j", version="2.14.1")
        assert entry.to_dict() == asdict(entry)


class TestReferenceEntryToDict:
    def test_to_dict_matches_asdict(self) -> None:
        entry = ReferenceEntry(url="https://nvd.nist.gov", source="nvd", tags=["Patch"])
        assert entry.to_dict() == asdict(entry)

    def test_to_dict_with_empty_tags(self) -> None:
        entry = ReferenceEntry(url="https://example.com")
        result = entry.to_dict()
        assert result["tags"] == []
