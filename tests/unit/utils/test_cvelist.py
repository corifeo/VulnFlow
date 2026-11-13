"""
cvelistV5 parsing utilities - map CVE JSON 5.x records to VulnFlow columns
Tests: CVSS/CWE/CPE/reference parsing, populate_cvelist_columns, release resolution

Layer: Utils
Dependencies: tests/fixtures/cvelist (real cvelistV5 records)
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from vulnflow.schema import CVSSv4Entry
from vulnflow.utils.cvelist import (
    parse_cvelist_cpes,
    parse_cvelist_cvss,
    parse_cvelist_cwes,
    parse_cvelist_references,
    populate_cvelist_columns,
)

CVELIST_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "cvelist"


def _record(cve_id: str) -> dict[str, Any]:
    return json.loads((CVELIST_DIR / f"{cve_id}.json").read_text(encoding="utf-8"))


def _containers(cve_id: str) -> dict[str, Any]:
    return _record(cve_id)["containers"]


class TestParseCvelistCvss:
    def test_cna_scores_authoritative_highest_version_first(self) -> None:
        scores = parse_cvelist_cvss(_containers("CVE-2025-0168"))

        assert [s.version for s in scores] == ["4.0", "3.1", "3.0", "2.0"]
        assert all(s.is_authoritative for s in scores)
        assert {s.source for s in scores} == {"vuldb"}
        assert isinstance(scores[0], CVSSv4Entry)
        assert scores[0].base_score == 5.3

    def test_v2_without_severity_gets_v2_range(self) -> None:
        v2 = parse_cvelist_cvss(_containers("CVE-2025-0168"))[-1]

        assert (v2.base_score, v2.base_severity) == (6.5, "MEDIUM")

    def test_adp_only_score_is_not_authoritative(self) -> None:
        scores = parse_cvelist_cvss(_containers("CVE-1999-0006"))

        assert len(scores) == 1
        assert scores[0].source == "cisa-adp"
        assert scores[0].is_authoritative is False
        assert (scores[0].base_score, scores[0].base_severity) == (9.8, "CRITICAL")
        assert scores[0].attack_vector == "NETWORK"

    def test_missing_v3_severity_computed_from_score(self) -> None:
        containers = {
            "cna": {
                "providerMetadata": {"shortName": "Acme"},
                "metrics": [{"cvssV3_1": {"version": "3.1", "baseScore": 9.1}}],
            }
        }

        (score,) = parse_cvelist_cvss(containers)

        assert (score.base_severity, score.source) == ("CRITICAL", "acme")

    def test_no_metrics_returns_empty(self) -> None:
        assert parse_cvelist_cvss(_containers("CVE-1999-0020")) == []


class TestParseCvelistCwes:
    def test_cna_cwes_in_order(self) -> None:
        cwes = parse_cvelist_cwes(_containers("CVE-2025-0168"))

        assert [c.id for c in cwes] == ["CWE-89", "CWE-74"]

    def test_adp_cwe_when_cna_has_none(self) -> None:
        cwes = parse_cvelist_cwes(_containers("CVE-1999-0006"))

        assert [c.id for c in cwes] == ["CWE-125"]

    def test_duplicates_dropped(self) -> None:
        pt = [{"descriptions": [{"cweId": "CWE-79"}, {"cweId": "CWE-79"}]}]
        containers = {"cna": {"problemTypes": pt}, "adp": [{"problemTypes": pt}]}

        assert [c.id for c in parse_cvelist_cwes(containers)] == ["CWE-79"]


class TestParseCvelistCpes:
    def test_cpe_applicability_keeps_fix_version(self) -> None:
        cpes = parse_cvelist_cpes(_containers("CVE-2005-10004"))

        cacti = [c for c in cpes if c.product == "cacti"]
        assert cacti and cacti[0].vendor == "cacti"
        assert cacti[0].fix_version == "0.8.6-d"

    def test_adp_affected_cpes(self) -> None:
        cpes = parse_cvelist_cpes(_containers("CVE-1999-0006"))

        assert [(c.vendor, c.product, c.version) for c in cpes] == [
            ("qualcomm", "qpopper", "2.4")
        ]

    def test_vendor_product_names_are_not_cpes(self) -> None:
        """CNA affected vendor/product names stay out of cpes (decision 2)."""
        assert parse_cvelist_cpes(_containers("CVE-2025-0168")) == []

    def test_cpe22_uri_parsed_and_duplicates_dropped(self) -> None:
        uri = "cpe:/o:redhat:enterprise_linux:4::as"
        containers = {"cna": {"affected": [{"cpes": [uri]}, {"cpes": [uri, "bad"]}]}}

        (cpe,) = parse_cvelist_cpes(containers)

        assert (cpe.vendor, cpe.product, cpe.version, cpe.raw) == (
            "redhat",
            "enterprise_linux",
            "4",
            uri,
        )


class TestParseCvelistReferences:
    def test_cna_and_adp_merged_by_url(self) -> None:
        refs = parse_cvelist_references(_containers("CVE-2002-0367"))

        urls = [r.url for r in refs]
        assert len(urls) == len(set(urls)) == 10
        assert refs[0].source == "mitre"
        assert refs[-1].source == "cisa-adp"

    def test_tags_kept(self) -> None:
        refs = parse_cvelist_references(_containers("CVE-1999-0006"))

        assert refs[0].tags == ["vendor-advisory", "x_refsource_SGI"]


class TestPopulateCvelistColumns:
    def test_empty_input_returns_empty_dict(self) -> None:
        assert populate_cvelist_columns([]) == {}

    def test_core_columns(self) -> None:
        arrays = populate_cvelist_columns([_record("CVE-2025-0168")])

        assert arrays["cve_id"][0] == "CVE-2025-0168"
        assert arrays["nvd_status"][0] == "published"
        assert arrays["description"][0].startswith("A vulnerability")
        assert len(json.loads(arrays["cvss_scores"][0])) == 4
        assert [c["id"] for c in json.loads(arrays["cwes"][0])] == ["CWE-89", "CWE-74"]
        assert arrays["kev"][0] == "null"
        assert arrays["epss_scores"][0] == "[]"
        meta = json.loads(arrays["metadata"][0])
        assert meta["source"] == "cvelistV5"
        assert meta["assigner"] == "VulDB"
        assert meta["state"] == "PUBLISHED"
        assert meta["date_reserved"].startswith("2025-01-01")
        assert "cve_valid" not in arrays

    def test_dates_match_cve_metadata(self) -> None:
        record = _record("CVE-2010-20103")

        arrays = populate_cvelist_columns([record])

        assert arrays["nvd_published_at"][0] == datetime(
            2025, 8, 20, 15, 38, 46, 697000, tzinfo=UTC
        )
        assert arrays["nvd_modified_at"][0] == datetime(
            2026, 7, 15, 1, 23, 11, 729000, tzinfo=UTC
        )

    def test_rejected_record(self) -> None:
        arrays = populate_cvelist_columns([_record("CVE-1999-0020")])

        assert arrays["nvd_status"][0] == "rejected"
        assert arrays["description"][0].startswith("DO NOT USE")
        assert arrays["cvss_scores"][0] == "[]"
        assert arrays["references"][0] == "[]"

    def test_ssvc_in_metadata(self) -> None:
        arrays = populate_cvelist_columns([_record("CVE-2002-0367")])

        ssvc = json.loads(arrays["metadata"][0])["ssvc"]
        assert {"Exploitation": "active"} in ssvc["options"]
        assert arrays["kev"][0] == "null"  # KEV comes from KEVEnricher

    def test_affected_in_metadata_with_purl(self) -> None:
        arrays = populate_cvelist_columns([_record("CVE-2010-20103")])

        (affected,) = json.loads(arrays["metadata"][0])["affected"]
        assert affected["vendor"] == "ProFTPD Project"
        assert affected["packageURL"] == "pkg:github/proftpd/proftpd"
        assert affected["versions"] == [{"status": "affected", "version": "1.3.3c"}]

    def test_english_description_chosen_when_not_first(self) -> None:
        record = _record("CVE-2005-10002")
        record["containers"]["cna"]["descriptions"].reverse()  # "de" first

        arrays = populate_cvelist_columns([record])

        assert arrays["description"][0].startswith("A vulnerability")

    def test_validation_columns_and_message(self) -> None:
        records = [_record("CVE-2025-0168"), _record("CVE-2022-0353")]
        validations: list[tuple[bool, str | None, str | None]] = [
            (True, None, None),
            (False, "FAILED_JSON_SCHEMA_VALIDATION", "must not be empty"),
        ]

        arrays = populate_cvelist_columns(records, validations)

        assert arrays["cve_valid"].dtype == bool
        assert list(arrays["cve_valid"]) == [True, False]
        assert list(arrays["cve_validation_error"]) == [
            None,
            "FAILED_JSON_SCHEMA_VALIDATION",
        ]
        assert "validation_message" not in json.loads(arrays["metadata"][0])
        meta = json.loads(arrays["metadata"][1])
        assert meta["validation_message"] == "must not be empty"

    def test_validations_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="validations"):
            populate_cvelist_columns([_record("CVE-2025-0168")], [])


def _release(tag: str) -> dict[str, Any]:
    """GitHub release shaped like cvelistV5's: midnight zip + cumulative delta."""
    day, hhmm = tag.removeprefix("cve_").split("_")
    base = f"https://github.com/CVEProject/cvelistV5/releases/download/{tag}"
    names = [
        f"{day}_all_CVEs_at_midnight.zip.zip",
        f"{day}_delta_CVEs_at_{hhmm}.zip",
        "release_notes.md",
    ]
    return {
        "tag_name": tag,
        "assets": [{"name": n, "browser_download_url": f"{base}/{n}"} for n in names],
    }


RELEASES = [  # newest first, as the API returns them
    _release("cve_2026-11-13_1700Z"),
    _release("cve_2026-11-13_1600Z"),
    _release("cve_2026-11-12_2300Z"),
]


class TestReleaseResolution:
    def test_latest_full_url(self) -> None:
        from vulnflow.utils.cvelist import latest_full_url

        assert latest_full_url(RELEASES) == (
            "https://github.com/CVEProject/cvelistV5/releases/download/"
            "cve_2026-11-13_1700Z/2026-11-13_all_CVEs_at_midnight.zip.zip"
        )

    def test_latest_full_url_without_asset_raises(self) -> None:
        from vulnflow.utils.cvelist import latest_full_url

        with pytest.raises(ValueError, match="midnight"):
            latest_full_url([{"tag_name": "x", "assets": []}])

    def test_latest_delta_url_for_day(self) -> None:
        from datetime import date

        from vulnflow.utils.cvelist import latest_delta_url

        today = latest_delta_url(list(reversed(RELEASES)), date(2026, 11, 13))
        yesterday = latest_delta_url(RELEASES, date(2026, 11, 12))

        assert today is not None and today.endswith(
            "2026-11-13_delta_CVEs_at_1700Z.zip"
        )
        assert yesterday is not None and yesterday.endswith("_delta_CVEs_at_2300Z.zip")
        assert latest_delta_url(RELEASES, date(2026, 11, 10)) is None

    def test_fetch_releases_uses_github_api(self) -> None:
        from unittest.mock import patch

        from vulnflow.utils.cvelist import RELEASES_URL, fetch_releases

        with patch("vulnflow.utils.cvelist.HttpFetcher") as MockFetcher:
            MockFetcher.return_value.fetch_sync.return_value = (
                json.dumps(RELEASES).encode(),
                "json",
            )
            releases = fetch_releases(timeout=5.0)

        assert releases == RELEASES
        url = MockFetcher.return_value.fetch_sync.call_args.args[0]
        assert url.startswith(RELEASES_URL)

    @pytest.mark.parametrize(
        ("label", "expected"),
        [
            ("CVEListExtractor(2026-11-13_all_CVEs_at_midnight.zip.zip)", "2026-11-13"),
            ("CVEListExtractor(2026-11-12_delta_CVEs_at_2300Z.zip)", "2026-11-12"),
            ("CVEListExtractor(cves)", None),
            ("NVDExtractor(nvdcve-2.0-2024)", None),
        ],
    )
    def test_snapshot_date(self, label: str, expected: str | None) -> None:
        from vulnflow.utils.cvelist import snapshot_date

        result = snapshot_date(label)

        assert (result.isoformat() if result else None) == expected
