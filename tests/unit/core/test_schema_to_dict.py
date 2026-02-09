"""Tests for schema dataclass to_dict() methods."""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime

from vulnflow.schema import CPEEntry, CVSSEntry, CWEEntry, ReferenceEntry


class TestCVSSEntryToDict:
    def test_to_dict_matches_asdict_simple(self) -> None:
        entry = CVSSEntry(
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
        entry = CVSSEntry(
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

    def test_to_dict_includes_all_fields(self) -> None:
        entry = CVSSEntry(
            version="3.1",
            vector_string="AV:N",
            base_score=5.0,
            base_severity="MEDIUM",
            source="nvd",
        )
        result = entry.to_dict()
        assert set(result.keys()) == {
            "version", "vector_string", "base_score", "base_severity",
            "source", "is_authoritative", "exploitability_score",
            "impact_score", "source_url", "published_at", "updated_at",
        }


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
