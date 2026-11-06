"""Tests for TriageLoader (SSVC triage queue export)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vulnflow.dataset import VulnDataset
from vulnflow.exceptions import LoaderError
from vulnflow.loaders import TRIAGE_SCHEMA, TriageLoader
from vulnflow.loaders.triage import (
    _title,
    _vector_metrics,
    suggest_automatable,
    suggest_exploitation,
    suggest_technical_impact,
)
from vulnflow.transforms.enrichers import EPSSEnricher, KEVEnricher
from vulnflow.transforms.flatten import (
    CPEFlattenTransform,
    CVSSFlattenTransform,
    EPSSFlattenTransform,
    KEVFlattenTransform,
)

FIXTURES = Path("tests/fixtures")


def _enriched(nvd: str, kev: str) -> VulnDataset:
    ds = VulnDataset.from_nvd(str(FIXTURES / nvd))
    ds.transform(
        [
            KEVEnricher(source=str(FIXTURES / kev)),
            EPSSEnricher(source=str(FIXTURES / "epss_sample.csv")),
            KEVFlattenTransform(),
            EPSSFlattenTransform(),
            CVSSFlattenTransform(),
            CPEFlattenTransform(),
        ]
    )
    return ds


@pytest.fixture
def sample() -> VulnDataset:
    return _enriched("nvd_sample.json", "kev_sample.csv")


@pytest.fixture
def log4shell() -> VulnDataset:
    return _enriched("nvd_log4shell.json", "kev_log4shell.csv")


# ---------------------------------------------------------------- heuristics


class TestExploitation:
    def test_kev_is_active(self) -> None:
        r = suggest_exploitation(True, "2021-12-10T00:00:00Z", [], None)
        assert r == {"value": "active", "reason": "Listed in CISA KEV on 2021-12-10"}

    def test_exploit_reference_is_poc(self) -> None:
        r = suggest_exploitation(False, None, ["https://x", "https://y"], None)
        assert r["value"] == "poc"
        assert "2 references" in r["reason"]

    def test_cvss4_attacked_is_active(self) -> None:
        assert suggest_exploitation(False, None, [], "A")["value"] == "active"

    def test_nothing_is_none(self) -> None:
        assert suggest_exploitation(False, None, [], None)["value"] == "none"


class TestAutomatable:
    def test_no_barriers_is_yes(self) -> None:
        m = _vector_metrics("CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H")
        assert suggest_automatable(m) == {
            "value": "yes",
            "reason": "AV:N, AC:L, PR:N, UI:N: no barrier to scripting at scale",
        }

    def test_user_interaction_is_barrier(self) -> None:
        m = _vector_metrics("CVSS:3.1/AV:N/AC:L/PR:N/UI:R/S:U/C:H/I:H/A:H")
        r = suggest_automatable(m)
        assert r is not None and r["value"] == "no" and "UI:R" in r["reason"]

    def test_cvss4_supplemental_wins(self) -> None:
        m = _vector_metrics("CVSS:4.0/AV:L/AC:H/AT:P/PR:H/UI:A/VC:H/VI:H/VA:H/AU:Y")
        r = suggest_automatable(m)
        assert r is not None and r["value"] == "yes"

    def test_no_vector(self) -> None:
        assert suggest_automatable({}) is None


class TestTechnicalImpact:
    def test_high_c_and_i_is_total(self) -> None:
        m = _vector_metrics("CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:N")
        r = suggest_technical_impact(m)
        assert r is not None and r["value"] == "total"

    def test_availability_only_is_partial(self) -> None:
        m = _vector_metrics("CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H")
        r = suggest_technical_impact(m)
        assert r is not None and r["value"] == "partial"

    def test_cvss4_uses_vulnerable_system_impact(self) -> None:
        m = _vector_metrics(
            "CVSS:4.0/AV:N/AC:L/AT:N/PR:N/UI:N/VC:H/VI:H/VA:H/SC:N/SI:N/SA:N"
        )
        r = suggest_technical_impact(m)
        assert r is not None and r["value"] == "total"


def test_title_prefers_kev_name() -> None:
    assert _title("CVE-1", "Log4j RCE", "Long text.") == "Log4j RCE"


def test_title_uses_first_sentence() -> None:
    assert _title("CVE-1", None, "First bit. Second bit.") == "First bit"


def test_title_falls_back_to_id() -> None:
    assert _title("CVE-1", None, None) == "CVE-1"


# ---------------------------------------------------------------- loader


def test_writes_schema_and_items(sample: VulnDataset, tmp_path: Path) -> None:
    out = tmp_path / "queue.json"
    meta = TriageLoader(out).load(sample)
    doc = json.loads(out.read_text())

    assert doc["schema"] == TRIAGE_SCHEMA
    assert meta["record_count"] == doc["count"] == len(doc["items"])
    assert doc["enrichment"] == {"kev": True, "epss": True, "cvss_flattened": True}
    for item in doc["items"]:
        assert item["id"].startswith("CVE-")
        assert item["evidence"]["exploitation"]["value"] in {"none", "poc", "active"}


def test_rejected_excluded(sample: VulnDataset) -> None:
    statuses = dict(
        zip(sample.get_column("cve_id"), sample.get_column("nvd_status"), strict=True)
    )
    ids = {i["id"] for i in TriageLoader("x.json").build(sample)["items"]}
    assert not any(statuses[i] == "rejected" for i in ids)


def test_ranked_kev_first_then_epss(sample: VulnDataset) -> None:
    items = TriageLoader("x.json").build(sample)["items"]
    kev_flags = [i["kev"]["listed"] for i in items]
    assert kev_flags == sorted(kev_flags, reverse=True)
    non_kev = [
        (i["epss"] or {}).get("probability") or 0
        for i in items
        if not i["kev"]["listed"]
    ]
    assert non_kev == sorted(non_kev, reverse=True)


def test_limit_and_matched(sample: VulnDataset) -> None:
    doc = TriageLoader("x.json", limit=1).build(sample)
    assert doc["count"] == 1
    assert doc["matched"] >= 1


def test_kev_only(sample: VulnDataset) -> None:
    items = TriageLoader("x.json", kev_only=True).build(sample)["items"]
    assert items and all(i["kev"]["listed"] for i in items)


def test_min_epss(sample: VulnDataset) -> None:
    items = TriageLoader("x.json", min_epss=0.5).build(sample)["items"]
    assert items and all(i["epss"]["probability"] >= 0.5 for i in items)


def test_cve_ids(sample: VulnDataset) -> None:
    items = TriageLoader("x.json", cve_ids=["CVE-2024-0002"]).build(sample)["items"]
    assert [i["id"] for i in items] == ["CVE-2024-0002"]


def test_bad_limit() -> None:
    with pytest.raises(LoaderError):
        TriageLoader("x.json", limit=0)


def test_empty_dataset(tmp_path: Path) -> None:
    doc = TriageLoader(tmp_path / "q.json").build(VulnDataset())
    assert doc["count"] == 0 and doc["items"] == []


def test_log4shell_evidence(log4shell: VulnDataset) -> None:
    (item,) = TriageLoader("x.json").build(log4shell)["items"]
    assert item["id"] == "CVE-2021-44228"
    assert item["title"] == "Apache Log4j2 Remote Code Execution Vulnerability"
    assert item["kev"]["listed"] is True and item["kev"]["ransomware"] is True
    assert item["cvss"]["metrics"]["AV"] == "N"
    ev = item["evidence"]
    assert ev["exploitation"]["value"] == "active"
    assert ev["automatable"]["value"] == "yes"
    assert ev["technical_impact"]["value"] == "total"
    assert item["exploit_refs"], "NVD tags Log4Shell references as Exploit"
    assert item["product"]["vendor"] == "Apache"
    assert "2.15.0" in item["product"]["fixed_in"]
    assert "2.7.0" not in item["product"]["fixed_in"], "Siemens firmware fix leaked"


def test_product_without_kev_uses_first_cpe() -> None:
    from vulnflow.loaders.triage import _product

    cpes = [
        {"vendor": "acme", "product": "widget", "fix_version": "1.2"},
        {"vendor": "other", "product": "thing", "fix_version": "9.9"},
    ]
    assert _product(None, None, cpes) == ("acme", "widget", ["1.2"])
