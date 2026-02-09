"""
Tag Inference - Infer tags from vulnerability data
Tests: all 7 tag categories, confidence thresholds, dict-based tags, missing data

Layer: Transform/Inferences
Dependencies: dataset, flatten transforms
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from vulnflow.dataset import VulnDataset
from vulnflow.transforms.inferences.tags import TagInference

# ============================================================================
# Helpers
# ============================================================================


def _make_dataset(columns: dict[str, list[Any]]) -> VulnDataset:
    """Create a VulnDataset from column lists."""
    dataset = VulnDataset()
    dataset._data = {k: np.array(v, dtype=object) for k, v in columns.items()}
    return dataset


def _get_tags(ds: VulnDataset, idx: int = 0) -> dict[str, list[str]]:
    """Parse inferred_tags column value at index as dict."""
    return json.loads(ds.get_column("inferred_tags")[idx])


# ============================================================================
# Registration & Schema
# ============================================================================


class TestTagInferenceRegistration:
    def test_creates_inferred_tags_column(self) -> None:
        """TagInference adds inferred_tags column to dataset."""
        ds = _make_dataset({"cve_id": ["CVE-2024-0001"], "description": [None]})
        TagInference().transform(ds)
        assert "inferred_tags" in ds.columns

    def test_output_is_json_encoded_dict(self) -> None:
        """Each value in inferred_tags is a JSON-encoded dict keyed by category."""
        ds = _make_dataset({
            "cve_id": ["CVE-2024-0001"],
            "description": ["A remote code execution vulnerability"],
        })
        TagInference().transform(ds)
        tags = _get_tags(ds)
        assert isinstance(tags, dict)
        assert "threat" in tags
        assert "rce" in tags["threat"]

    def test_schema_version_is_3(self) -> None:
        """SCHEMA_VERSION is 3.0.0 (breaking: tier system added)."""
        assert TagInference.SCHEMA_VERSION == "3.0.0"

    def test_depends_on_flatten_transforms(self) -> None:
        """TagInference depends on all 5 flatten transforms."""
        deps = TagInference.depends_on
        assert "KEVFlattenTransform" in deps
        assert "EPSSFlattenTransform" in deps
        assert "CWEFlattenTransform" in deps
        assert "CVSSFlattenTransform" in deps
        assert "CPEFlattenTransform" in deps


# ============================================================================
# Threat Tags (from description)
# ============================================================================


class TestThreatTags:
    def test_rce_from_description(self) -> None:
        """Detect 'remote code execution' keyword."""
        ds = _make_dataset({
            "cve_id": ["CVE-2024-0001"],
            "description": ["allows remote code execution via crafted request"],
        })
        TagInference().transform(ds)
        assert "rce" in _get_tags(ds)["threat"]

    def test_sqli_from_description(self) -> None:
        """Detect 'sql injection' keyword."""
        ds = _make_dataset({
            "cve_id": ["CVE-2024-0001"],
            "description": ["SQL injection vulnerability in login page"],
        })
        TagInference().transform(ds)
        assert "sqli" in _get_tags(ds)["threat"]

    def test_xss_from_description(self) -> None:
        """Detect cross-site scripting variants."""
        ds = _make_dataset({
            "cve_id": ["CVE-2024-0001"],
            "description": ["Cross-site scripting (XSS) in the admin panel"],
        })
        TagInference().transform(ds)
        assert "xss" in _get_tags(ds)["threat"]

    def test_multiple_threats_from_description(self) -> None:
        """Multiple threat keywords produce multiple tags."""
        ds = _make_dataset({
            "cve_id": ["CVE-2024-0001"],
            "description": ["SQL injection leading to remote code execution"],
        })
        TagInference().transform(ds)
        threats = _get_tags(ds)["threat"]
        assert "sqli" in threats
        assert "rce" in threats

    def test_auth_bypass_from_description(self) -> None:
        """Detect 'authentication bypass' keyword."""
        ds = _make_dataset({
            "cve_id": ["CVE-2024-0001"],
            "description": ["authentication bypass in OAuth flow"],
        })
        TagInference().transform(ds)
        assert "auth-bypass" in _get_tags(ds)["threat"]

    def test_ssrf_from_description(self) -> None:
        """Detect SSRF keyword."""
        ds = _make_dataset({
            "cve_id": ["CVE-2024-0001"],
            "description": ["Server-side request forgery allows internal network access"],
        })
        TagInference().transform(ds)
        assert "ssrf" in _get_tags(ds)["threat"]

    def test_info_disclosure_from_description(self) -> None:
        """Detect information disclosure keyword."""
        ds = _make_dataset({
            "cve_id": ["CVE-2024-0001"],
            "description": ["information disclosure via error messages"],
        })
        TagInference().transform(ds)
        assert "info-disclosure" in _get_tags(ds)["threat"]

    def test_no_tags_from_empty_description(self) -> None:
        """Empty description produces no threat tags."""
        ds = _make_dataset({
            "cve_id": ["CVE-2024-0001"],
            "description": [None],
        })
        TagInference().transform(ds)
        assert "threat" not in _get_tags(ds)

    def test_no_tags_from_generic_description(self) -> None:
        """Generic description without keywords produces no threat tags."""
        ds = _make_dataset({
            "cve_id": ["CVE-2024-0001"],
            "description": ["A vulnerability exists in the software."],
        })
        TagInference().transform(ds)
        assert "threat" not in _get_tags(ds)


# ============================================================================
# Vector Tags (from CWE)
# ============================================================================


class TestVectorTags:
    def test_web_vector_from_xss_cwe(self) -> None:
        """CWE-79 (XSS) maps to vector:web."""
        ds = _make_dataset({
            "cve_id": ["CVE-2024-0001"],
            "description": [None],
            "cwes": [json.dumps([{"id": "CWE-79"}])],
        })
        TagInference().transform(ds)
        assert "web" in _get_tags(ds)["vector"]

    def test_memory_vector_from_buffer_cwe(self) -> None:
        """CWE-119 (Buffer Overflow) maps to vector:memory."""
        ds = _make_dataset({
            "cve_id": ["CVE-2024-0001"],
            "description": [None],
            "cwes": [json.dumps([{"id": "CWE-119"}])],
        })
        TagInference().transform(ds)
        assert "memory" in _get_tags(ds)["vector"]

    def test_auth_vector_from_cwe_287(self) -> None:
        """CWE-287 (Improper Authentication) maps to vector:authentication."""
        ds = _make_dataset({
            "cve_id": ["CVE-2024-0001"],
            "description": [None],
            "cwes": [json.dumps([{"id": "CWE-287"}])],
        })
        TagInference().transform(ds)
        assert "authentication" in _get_tags(ds)["vector"]

    def test_no_vector_from_unmapped_cwe(self) -> None:
        """Unknown CWE produces no vector tags."""
        ds = _make_dataset({
            "cve_id": ["CVE-2024-0001"],
            "description": [None],
            "cwes": [json.dumps([{"id": "CWE-9999"}])],
        })
        TagInference().transform(ds)
        assert "vector" not in _get_tags(ds)

    def test_no_vector_from_missing_cwes(self) -> None:
        """Missing CWE column produces no vector tags."""
        ds = _make_dataset({
            "cve_id": ["CVE-2024-0001"],
            "description": [None],
        })
        TagInference().transform(ds)
        assert "vector" not in _get_tags(ds)


# ============================================================================
# Product Tags (from CPE)
# ============================================================================


class TestProductTags:
    def test_web_server_from_apache(self) -> None:
        """Apache HTTP Server maps to product:web-server."""
        ds = _make_dataset({
            "cve_id": ["CVE-2024-0001"],
            "description": [None],
            "cpes": [json.dumps([{"vendor": "apache", "product": "http_server"}])],
        })
        TagInference().transform(ds)
        assert "web-server" in _get_tags(ds)["product"]

    def test_database_from_mysql(self) -> None:
        """MySQL maps to product:database."""
        ds = _make_dataset({
            "cve_id": ["CVE-2024-0001"],
            "description": [None],
            "cpes": [json.dumps([{"vendor": "mysql", "product": "mysql"}])],
        })
        TagInference().transform(ds)
        assert "database" in _get_tags(ds)["product"]

    def test_browser_from_google_chrome(self) -> None:
        """Google Chrome maps to product:browser."""
        ds = _make_dataset({
            "cve_id": ["CVE-2024-0001"],
            "description": [None],
            "cpes": [json.dumps([{"vendor": "google", "product": "chrome"}])],
        })
        TagInference().transform(ds)
        assert "browser" in _get_tags(ds)["product"]

    def test_no_product_from_unknown_vendor(self) -> None:
        """Unknown vendor/product produces no product tags."""
        ds = _make_dataset({
            "cve_id": ["CVE-2024-0001"],
            "description": [None],
            "cpes": [json.dumps([{"vendor": "unknown_co", "product": "unknown_app"}])],
        })
        TagInference().transform(ds)
        assert "product" not in _get_tags(ds)


# ============================================================================
# Attack Surface Tags (from CVSS vector)
# ============================================================================


class TestAttackSurfaceTags:
    def test_network_attack_vector(self) -> None:
        """CVSS AV:N maps to attack:network."""
        ds = _make_dataset({
            "cve_id": ["CVE-2024-0001"],
            "description": [None],
            "cvss_vector_string": ["CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"],
        })
        TagInference().transform(ds)
        assert "network" in _get_tags(ds)["attack"]

    def test_local_attack_vector(self) -> None:
        """CVSS AV:L maps to attack:local."""
        ds = _make_dataset({
            "cve_id": ["CVE-2024-0001"],
            "description": [None],
            "cvss_vector_string": ["CVSS:3.1/AV:L/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"],
        })
        TagInference().transform(ds)
        assert "local" in _get_tags(ds)["attack"]

    def test_no_auth_required(self) -> None:
        """CVSS PR:N maps to attack:no-auth."""
        ds = _make_dataset({
            "cve_id": ["CVE-2024-0001"],
            "description": [None],
            "cvss_vector_string": ["CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"],
        })
        TagInference().transform(ds)
        assert "no-auth" in _get_tags(ds)["attack"]

    def test_no_user_interaction(self) -> None:
        """CVSS UI:N maps to attack:no-interaction."""
        ds = _make_dataset({
            "cve_id": ["CVE-2024-0001"],
            "description": [None],
            "cvss_vector_string": ["CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"],
        })
        TagInference().transform(ds)
        assert "no-interaction" in _get_tags(ds)["attack"]

    def test_requires_interaction(self) -> None:
        """CVSS UI:R maps to attack:requires-interaction."""
        ds = _make_dataset({
            "cve_id": ["CVE-2024-0001"],
            "description": [None],
            "cvss_vector_string": ["CVSS:3.1/AV:N/AC:L/PR:N/UI:R/S:U/C:H/I:H/A:H"],
        })
        TagInference().transform(ds)
        assert "requires-interaction" in _get_tags(ds)["attack"]

    def test_no_attack_tags_from_missing_cvss(self) -> None:
        """Missing CVSS vector produces no attack tags."""
        ds = _make_dataset({
            "cve_id": ["CVE-2024-0001"],
            "description": [None],
        })
        TagInference().transform(ds)
        assert "attack" not in _get_tags(ds)


# ============================================================================
# Exploitation Tags (from KEV, EPSS, exploits)
# ============================================================================


class TestExploitationTags:
    def test_actively_exploited_from_kev(self) -> None:
        """is_kev=True maps to status:actively-exploited."""
        ds = _make_dataset({
            "cve_id": ["CVE-2024-0001"],
            "description": [None],
            "is_kev": [True],
        })
        TagInference().transform(ds)
        assert "actively-exploited" in _get_tags(ds)["status"]

    def test_ransomware_from_kev(self) -> None:
        """kev_ransomware=True maps to status:ransomware."""
        ds = _make_dataset({
            "cve_id": ["CVE-2024-0001"],
            "description": [None],
            "is_kev": [True],
            "kev_ransomware": [True],
        })
        TagInference().transform(ds)
        assert "ransomware" in _get_tags(ds)["status"]

    def test_high_epss_from_probability(self) -> None:
        """epss_probability >= 0.5 maps to status:high-epss."""
        ds = _make_dataset({
            "cve_id": ["CVE-2024-0001"],
            "description": [None],
            "epss_probability": [0.75],
        })
        TagInference().transform(ds)
        assert "high-epss" in _get_tags(ds)["status"]

    def test_moderate_epss_from_probability(self) -> None:
        """epss_probability >= 0.1 and < 0.5 maps to status:moderate-epss."""
        ds = _make_dataset({
            "cve_id": ["CVE-2024-0001"],
            "description": [None],
            "epss_probability": [0.25],
        })
        TagInference().transform(ds)
        status = _get_tags(ds)["status"]
        assert "moderate-epss" in status
        assert "high-epss" not in status

    def test_has_public_exploit(self) -> None:
        """Exploit entries map to status:has-public-exploit."""
        ds = _make_dataset({
            "cve_id": ["CVE-2024-0001"],
            "description": [None],
            "exploits": [json.dumps([{"source": "exploitdb", "id": "12345"}])],
        })
        TagInference().transform(ds)
        assert "has-public-exploit" in _get_tags(ds)["status"]

    def test_no_exploitation_tags_from_missing_data(self) -> None:
        """Missing KEV/EPSS/exploits produces no status tags."""
        ds = _make_dataset({
            "cve_id": ["CVE-2024-0001"],
            "description": [None],
        })
        TagInference().transform(ds)
        assert "status" not in _get_tags(ds)

    def test_not_kev_produces_no_active_exploit_tag(self) -> None:
        """is_kev=False does not add status:actively-exploited."""
        ds = _make_dataset({
            "cve_id": ["CVE-2024-0001"],
            "description": [None],
            "is_kev": [False],
        })
        TagInference().transform(ds)
        tags = _get_tags(ds)
        if "status" in tags:
            assert "actively-exploited" not in tags["status"]


# ============================================================================
# Reference Tags (from NVD reference tags)
# ============================================================================


class TestReferenceTags:
    def test_has_patch_from_reference(self) -> None:
        """Reference with 'Patch' tag maps to ref:has-patch."""
        ds = _make_dataset({
            "cve_id": ["CVE-2024-0001"],
            "description": [None],
            "references": [json.dumps([
                {"url": "https://example.com/patch", "tags": ["Patch"]}
            ])],
        })
        TagInference().transform(ds)
        assert "has-patch" in _get_tags(ds)["ref"]

    def test_has_advisory_from_reference(self) -> None:
        """Reference with 'Vendor Advisory' tag maps to ref:has-advisory."""
        ds = _make_dataset({
            "cve_id": ["CVE-2024-0001"],
            "description": [None],
            "references": [json.dumps([
                {"url": "https://vendor.com/sa-001", "tags": ["Vendor Advisory"]}
            ])],
        })
        TagInference().transform(ds)
        assert "has-advisory" in _get_tags(ds)["ref"]

    def test_has_exploit_from_reference(self) -> None:
        """Reference with 'Exploit' tag maps to ref:has-exploit."""
        ds = _make_dataset({
            "cve_id": ["CVE-2024-0001"],
            "description": [None],
            "references": [json.dumps([
                {"url": "https://exploit-db.com/1234", "tags": ["Exploit"]}
            ])],
        })
        TagInference().transform(ds)
        assert "has-exploit" in _get_tags(ds)["ref"]

    def test_has_mitigation_from_reference(self) -> None:
        """Reference with 'Mitigation' tag maps to ref:has-mitigation."""
        ds = _make_dataset({
            "cve_id": ["CVE-2024-0001"],
            "description": [None],
            "references": [json.dumps([
                {"url": "https://example.com/workaround", "tags": ["Mitigation"]}
            ])],
        })
        TagInference().transform(ds)
        assert "has-mitigation" in _get_tags(ds)["ref"]

    def test_govt_tracked_from_reference(self) -> None:
        """Reference with 'US Government Resource' tag maps to ref:govt-tracked."""
        ds = _make_dataset({
            "cve_id": ["CVE-2024-0001"],
            "description": [None],
            "references": [json.dumps([
                {"url": "https://us-cert.cisa.gov/alert", "tags": ["US Government Resource"]}
            ])],
        })
        TagInference().transform(ds)
        assert "govt-tracked" in _get_tags(ds)["ref"]

    def test_multiple_reference_tags(self) -> None:
        """Multiple reference tags produce multiple ref: tags."""
        ds = _make_dataset({
            "cve_id": ["CVE-2024-0001"],
            "description": [None],
            "references": [json.dumps([
                {"url": "https://example.com/patch", "tags": ["Patch"]},
                {"url": "https://vendor.com/sa-001", "tags": ["Vendor Advisory"]},
            ])],
        })
        TagInference().transform(ds)
        ref = _get_tags(ds)["ref"]
        assert "has-patch" in ref
        assert "has-advisory" in ref

    def test_no_ref_tags_from_missing_references(self) -> None:
        """Missing references column produces no ref tags."""
        ds = _make_dataset({
            "cve_id": ["CVE-2024-0001"],
            "description": [None],
        })
        TagInference().transform(ds)
        assert "ref" not in _get_tags(ds)


# ============================================================================
# Impact Tags (from CVSS + description)
# ============================================================================


class TestImpactTags:
    def test_confidentiality_impact_from_cvss(self) -> None:
        """CVSS C:H maps to impact:confidentiality."""
        ds = _make_dataset({
            "cve_id": ["CVE-2024-0001"],
            "description": [None],
            "cvss_vector_string": ["CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:N/A:N"],
        })
        TagInference().transform(ds)
        assert "confidentiality" in _get_tags(ds)["impact"]

    def test_integrity_impact_from_cvss(self) -> None:
        """CVSS I:H maps to impact:integrity."""
        ds = _make_dataset({
            "cve_id": ["CVE-2024-0001"],
            "description": [None],
            "cvss_vector_string": ["CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:H/A:N"],
        })
        TagInference().transform(ds)
        assert "integrity" in _get_tags(ds)["impact"]

    def test_availability_impact_from_cvss(self) -> None:
        """CVSS A:H maps to impact:availability."""
        ds = _make_dataset({
            "cve_id": ["CVE-2024-0001"],
            "description": [None],
            "cvss_vector_string": ["CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H"],
        })
        TagInference().transform(ds)
        assert "availability" in _get_tags(ds)["impact"]

    def test_code_exec_impact_from_description(self) -> None:
        """Description 'execute arbitrary code' maps to impact:code-exec."""
        ds = _make_dataset({
            "cve_id": ["CVE-2024-0001"],
            "description": ["allows an attacker to execute arbitrary code"],
        })
        TagInference().transform(ds)
        assert "code-exec" in _get_tags(ds)["impact"]

    def test_data_leak_impact_from_description(self) -> None:
        """Description 'sensitive data' maps to impact:data-leak."""
        ds = _make_dataset({
            "cve_id": ["CVE-2024-0001"],
            "description": ["disclosure of sensitive data to unauthorized actor"],
        })
        TagInference().transform(ds)
        assert "data-leak" in _get_tags(ds)["impact"]

    def test_no_impact_from_low_cvss_scores(self) -> None:
        """CVSS C:N/I:N/A:N produces no impact tags from CVSS."""
        ds = _make_dataset({
            "cve_id": ["CVE-2024-0001"],
            "description": [None],
            "cvss_vector_string": ["CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:N"],
        })
        TagInference().transform(ds)
        assert "impact" not in _get_tags(ds)


# ============================================================================
# Confidence Threshold
# ============================================================================


class TestConfidenceThreshold:
    def test_high_threshold_filters_moderate_confidence(self) -> None:
        """Threshold 1.0 excludes 0.9-confidence reference tags."""
        ds = _make_dataset({
            "cve_id": ["CVE-2024-0001"],
            "description": [None],
            "references": [json.dumps([
                {"url": "https://example.com/patch", "tags": ["Patch"]}
            ])],
        })
        TagInference(confidence_threshold=1.0).transform(ds)
        # Reference tags have 0.9 confidence, should be filtered
        assert "ref" not in _get_tags(ds)

    def test_low_threshold_includes_all(self) -> None:
        """Threshold 0.1 includes all tags."""
        ds = _make_dataset({
            "cve_id": ["CVE-2024-0001"],
            "description": ["remote code execution"],
            "references": [json.dumps([
                {"url": "https://example.com/patch", "tags": ["Patch"]}
            ])],
        })
        TagInference(confidence_threshold=0.1).transform(ds)
        tags = _get_tags(ds)
        assert "rce" in tags["threat"]
        assert "has-patch" in tags["ref"]

    def test_default_threshold_is_0_7(self) -> None:
        """Default confidence threshold is 0.7."""
        ti = TagInference()
        assert ti.confidence_threshold == 0.7


# ============================================================================
# Deduplication & Sorting
# ============================================================================


class TestDeduplication:
    def test_tags_are_deduplicated(self) -> None:
        """Duplicate tags from different rules are deduplicated."""
        ds = _make_dataset({
            "cve_id": ["CVE-2024-0001"],
            "description": [None],
            "cwes": [json.dumps([{"id": "CWE-79"}, {"id": "CWE-89"}])],
        })
        TagInference().transform(ds)
        # vector:web should appear only once even if from multiple CWEs
        assert _get_tags(ds)["vector"].count("web") == 1

    def test_tags_within_category_are_sorted(self) -> None:
        """Tags within each category are sorted alphabetically."""
        ds = _make_dataset({
            "cve_id": ["CVE-2024-0001"],
            "description": ["remote code execution via SQL injection"],
            "cwes": [json.dumps([{"id": "CWE-79"}])],
        })
        TagInference().transform(ds)
        tags = _get_tags(ds)
        for values in tags.values():
            assert values == sorted(values)


# ============================================================================
# Multiple Records
# ============================================================================


class TestMultipleRecords:
    def test_each_record_tagged_independently(self) -> None:
        """Each record gets its own tag set."""
        ds = _make_dataset({
            "cve_id": ["CVE-2024-0001", "CVE-2024-0002"],
            "description": [
                "remote code execution vulnerability",
                "SQL injection in login form",
            ],
        })
        TagInference().transform(ds)
        tags0 = _get_tags(ds, 0)
        tags1 = _get_tags(ds, 1)
        assert "rce" in tags0["threat"]
        assert "threat" not in tags1 or "rce" not in tags1.get("threat", [])
        assert "sqli" in tags1["threat"]
        assert "threat" not in tags0 or "sqli" not in tags0.get("threat", [])

    def test_empty_dataset(self) -> None:
        """Empty dataset produces empty inferred_tags column."""
        ds = VulnDataset()
        ds._data = {"cve_id": np.array([], dtype=object)}
        TagInference().transform(ds)
        tags = ds.get_column("inferred_tags")
        assert len(tags) == 0


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    def test_malformed_json_cwes_skipped(self) -> None:
        """Malformed CWE JSON is silently skipped."""
        ds = _make_dataset({
            "cve_id": ["CVE-2024-0001"],
            "description": [None],
            "cwes": ["not valid json"],
        })
        TagInference().transform(ds)
        assert "vector" not in _get_tags(ds)

    def test_malformed_json_references_skipped(self) -> None:
        """Malformed references JSON is silently skipped."""
        ds = _make_dataset({
            "cve_id": ["CVE-2024-0001"],
            "description": [None],
            "references": ["{{bad json"],
        })
        TagInference().transform(ds)
        assert "ref" not in _get_tags(ds)

    def test_none_epss_probability_skipped(self) -> None:
        """None epss_probability produces no status tags."""
        ds = _make_dataset({
            "cve_id": ["CVE-2024-0001"],
            "description": [None],
            "epss_probability": [None],
        })
        TagInference().transform(ds)
        assert "status" not in _get_tags(ds)

    def test_invalid_cvss_vector_skipped(self) -> None:
        """Invalid CVSS vector string is silently skipped."""
        ds = _make_dataset({
            "cve_id": ["CVE-2024-0001"],
            "description": [None],
            "cvss_vector_string": ["not-a-vector"],
        })
        TagInference().transform(ds)
        assert "attack" not in _get_tags(ds)

    def test_all_data_sources_combined(self) -> None:
        """Full record with all data sources produces tags from all categories."""
        ds = _make_dataset({
            "cve_id": ["CVE-2024-0001"],
            "description": ["remote code execution vulnerability"],
            "cwes": [json.dumps([{"id": "CWE-79"}])],
            "cpes": [json.dumps([{"vendor": "apache", "product": "http_server"}])],
            "cvss_vector_string": ["CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"],
            "is_kev": [True],
            "kev_ransomware": [False],
            "epss_probability": [0.85],
            "exploits": [json.dumps([{"source": "exploitdb", "id": "999"}])],
            "references": [json.dumps([
                {"url": "https://example.com/patch", "tags": ["Patch"]}
            ])],
        })
        TagInference().transform(ds)
        tags = _get_tags(ds)

        # Should have keys for all 7 categories
        assert "threat" in tags
        assert "vector" in tags
        assert "product" in tags
        assert "attack" in tags
        assert "status" in tags
        assert "ref" in tags
        assert "impact" in tags


# ============================================================================
# Standard Pattern: Overwrite & Idempotency
# ============================================================================


class TestOverwrite:
    def test_overwrite_false_preserves_existing(self) -> None:
        """overwrite=False skips records with existing inferred_tags."""
        ds = _make_dataset({
            "cve_id": ["CVE-2024-0001"],
            "description": ["remote code execution vulnerability"],
            "inferred_tags": [json.dumps({"custom": ["my-tag"]})],
        })
        TagInference(overwrite=False).transform(ds)
        tags = _get_tags(ds)
        # Should preserve existing value, not overwrite
        assert tags == {"custom": ["my-tag"]}

    def test_overwrite_true_replaces(self) -> None:
        """overwrite=True replaces existing inferred_tags."""
        ds = _make_dataset({
            "cve_id": ["CVE-2024-0001"],
            "description": ["remote code execution vulnerability"],
            "inferred_tags": [json.dumps({"custom": ["my-tag"]})],
        })
        TagInference(overwrite=True).transform(ds)
        tags = _get_tags(ds)
        # Should have re-inferred, not the custom value
        assert "threat" in tags
        assert "rce" in tags["threat"]

    def test_overwrite_false_is_default(self) -> None:
        """Default overwrite is False."""
        ti = TagInference()
        assert ti.overwrite is False

    def test_idempotent_double_run(self) -> None:
        """Running twice without overwrite produces same result."""
        ds = _make_dataset({
            "cve_id": ["CVE-2024-0001"],
            "description": ["remote code execution vulnerability"],
        })
        TagInference().transform(ds)
        first = ds.get_column("inferred_tags")[0]

        TagInference().transform(ds)
        second = ds.get_column("inferred_tags")[0]

        assert first == second

    def test_overwrite_false_processes_none_records(self) -> None:
        """overwrite=False still processes records where inferred_tags is None."""
        ds = _make_dataset({
            "cve_id": ["CVE-2024-0001", "CVE-2024-0002"],
            "description": [
                "remote code execution",
                "SQL injection vulnerability",
            ],
            "inferred_tags": [
                json.dumps({"custom": ["keep-me"]}),
                None,
            ],
        })
        TagInference(overwrite=False).transform(ds)
        # First record preserved
        tags0 = _get_tags(ds, 0)
        assert tags0 == {"custom": ["keep-me"]}
        # Second record processed
        tags1 = _get_tags(ds, 1)
        assert "sqli" in tags1["threat"]


# ============================================================================
# Standard Pattern: Tier System
# ============================================================================


class TestTierSystem:
    def test_tier_thresholds_defined(self) -> None:
        """TagInference has TIER_THRESHOLDS class attribute."""
        assert hasattr(TagInference, "TIER_THRESHOLDS")
        assert isinstance(TagInference.TIER_THRESHOLDS, dict)
        assert len(TagInference.TIER_THRESHOLDS) == 3

    def test_tier1_only(self) -> None:
        """tiers=[1] only runs tier 1 rules (threat/vector/product/attack)."""
        ds = _make_dataset({
            "cve_id": ["CVE-2024-0001"],
            "description": ["remote code execution vulnerability"],
            "references": [json.dumps([
                {"url": "https://example.com/patch", "tags": ["Patch"]}
            ])],
            "epss_probability": [0.75],
            "is_kev": [True],
        })
        TagInference(tiers=[1]).transform(ds)
        tags = _get_tags(ds)
        # Tier 1 includes threat tags from description
        assert "threat" in tags
        # Tier 2 ref tags should NOT be present
        assert "ref" not in tags
        # Tier 3 epss tags should NOT be present
        assert "status" not in tags or "high-epss" not in tags.get("status", [])

    def test_tier2_excluded(self) -> None:
        """tiers=[1, 3] excludes tier 2 (reference/exploitation tags)."""
        ds = _make_dataset({
            "cve_id": ["CVE-2024-0001"],
            "description": ["remote code execution vulnerability"],
            "references": [json.dumps([
                {"url": "https://example.com/patch", "tags": ["Patch"]}
            ])],
            "is_kev": [True],
        })
        TagInference(tiers=[1, 3]).transform(ds)
        tags = _get_tags(ds)
        assert "threat" in tags  # tier 1
        assert "ref" not in tags  # tier 2 excluded

    def test_tier3_excluded(self) -> None:
        """tiers=[1, 2] excludes tier 3 (EPSS/impact heuristic tags)."""
        ds = _make_dataset({
            "cve_id": ["CVE-2024-0001"],
            "description": ["allows execute arbitrary code"],
            "epss_probability": [0.75],
            "references": [json.dumps([
                {"url": "https://example.com/patch", "tags": ["Patch"]}
            ])],
        })
        TagInference(tiers=[1, 2]).transform(ds)
        tags = _get_tags(ds)
        assert "ref" in tags  # tier 2
        # Tier 3 EPSS tags should NOT be present
        assert "status" not in tags or "high-epss" not in tags.get("status", [])

    def test_explicit_tier_selection(self) -> None:
        """Explicit tiers override threshold gating for tier selection."""
        ds = _make_dataset({
            "cve_id": ["CVE-2024-0001"],
            "description": ["remote code execution vulnerability"],
            "epss_probability": [0.75],
        })
        # threshold=0.85 would exclude tier 3 (0.8) by threshold gating,
        # but explicit tiers=[3] overrides the tier gate
        TagInference(confidence_threshold=0.85, tiers=[3]).transform(ds)
        tags = _get_tags(ds)
        assert "status" in tags
        assert "high-epss" in tags["status"]

    def test_should_run_inherited(self) -> None:
        """_should_run is inherited from BaseInference."""
        from vulnflow.transforms.base import BaseInference

        assert "_should_run" not in TagInference.__dict__
        assert TagInference._should_run is BaseInference._should_run


# ============================================================================
# Standard Pattern: Schema Version
# ============================================================================


class TestSchemaVersion:
    def test_schema_version_is_3(self) -> None:
        """SCHEMA_VERSION is 3.0.0 after tier system addition."""
        assert TagInference.SCHEMA_VERSION == "3.0.0"
