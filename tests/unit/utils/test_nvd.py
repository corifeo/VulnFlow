"""
NVD Parsing Utilities - parse_timestamp, extract_nvd_status, parse_cvss, etc.

Layer: Utils
Dependencies: schema, utils.cvss, utils.cpe
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
import pytest

from vulnflow.schema import CVSSv3Entry, CVSSv4Entry
from vulnflow.utils.nvd import (
    extract_nvd_status,
    parse_cpes,
    parse_cvss,
    parse_cwes,
    parse_references,
    parse_timestamp,
    to_utc_datetime,
)

# --- to_utc_datetime ---


@pytest.mark.parametrize(
    "value", [None, "", float("nan"), np.datetime64("NaT", "us"), pd.NaT]
)
def test_to_utc_datetime_nulls_return_none(value: Any) -> None:
    assert to_utc_datetime(value) is None


@pytest.mark.parametrize(
    "value",
    [
        "2021-12-10T00:00:00+00:00",
        "2021-12-10T00:00:00Z",
        "2021-12-10T00:00:00",
        "2021-12-10T01:00:00+01:00",
        datetime(2021, 12, 10),
        datetime(2021, 12, 10, tzinfo=UTC),
        np.datetime64("2021-12-10"),
        pd.Timestamp("2021-12-10", tz="UTC"),
    ],
)
def test_to_utc_datetime_all_forms_agree(value: Any) -> None:
    result = to_utc_datetime(value)
    assert result == datetime(2021, 12, 10, tzinfo=UTC)
    assert result is not None and result.utcoffset().total_seconds() == 0


def test_to_utc_datetime_malformed_string_raises() -> None:
    with pytest.raises(ValueError):
        to_utc_datetime("not-a-date")


def test_to_utc_datetime_unknown_type_raises() -> None:
    with pytest.raises(TypeError, match="int"):
        to_utc_datetime(123)


# --- Helpers ---


def _make_nvd_metric(
    version: str = "3.1",
    key_suffix: str = "V31",
    base_score: float = 9.8,
    base_severity: str | None = "CRITICAL",
    source: str = "nvd@nist.gov",
    metric_type: str = "Primary",
    vector_string: str = "",
    severity_at_metric_level: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """Build an NVD metrics dict for a single CVSS entry.

    Args:
        severity_at_metric_level: If True, puts baseSeverity on the metric
            (v2.0 style) instead of in cvssData. If base_severity is None,
            omits it entirely (v4.0 style).
    """
    cvss_data: dict[str, Any] = {
        "version": version,
        "vectorString": vector_string,
        "baseScore": base_score,
    }
    metric: dict[str, Any] = {
        "source": source,
        "type": metric_type,
        "cvssData": cvss_data,
    }

    if base_severity is not None:
        if severity_at_metric_level:
            metric["baseSeverity"] = base_severity
        else:
            cvss_data["baseSeverity"] = base_severity

    return {f"cvssMetric{key_suffix}": [metric]}


# --- parse_timestamp ---


def test_parse_timestamp_valid() -> None:
    """Parse ISO timestamp with Z suffix."""
    result = parse_timestamp("2024-01-10T10:00:00.000Z")
    assert result is not None
    assert result.year == 2024
    assert result.month == 1
    assert result.day == 10


def test_parse_timestamp_none() -> None:
    """None and empty string return None."""
    assert parse_timestamp(None) is None
    assert parse_timestamp("") is None


# --- extract_nvd_status ---


@pytest.mark.parametrize(
    ("vuln_status", "expected"),
    [
        ("Analyzed", "analyzed"),
        ("Modified", "modified"),
        ("Rejected", "rejected"),
        ("Deferred", "deferred"),
    ],
)
def test_extract_nvd_status_direct_mapping(vuln_status: str, expected: str) -> None:
    """Direct vulnStatus values map to normalized form."""
    assert extract_nvd_status({"vulnStatus": vuln_status}) == expected


def test_extract_nvd_status_awaiting_variants() -> None:
    """Awaiting, Received, Undergoing all map to 'awaiting'."""
    assert extract_nvd_status({"vulnStatus": "Awaiting Analysis"}) == "awaiting"
    assert extract_nvd_status({"vulnStatus": "Received"}) == "awaiting"
    assert extract_nvd_status({"vulnStatus": "Undergoing Analysis"}) == "awaiting"


def test_extract_nvd_status_missing_defaults_analyzed() -> None:
    """Missing vulnStatus defaults to analyzed."""
    assert (
        extract_nvd_status({"descriptions": [{"value": "Normal vuln"}]}) == "analyzed"
    )


def test_extract_nvd_status_missing_rejected_description() -> None:
    """Missing vulnStatus with REJECTED description -> rejected."""
    cve_data = {"descriptions": [{"value": "** REJECTED ** Duplicate"}]}
    assert extract_nvd_status(cve_data) == "rejected"


# --- parse_cvss ---


def test_parse_cvss_v40_computes_severity() -> None:
    """CVSS 4.0: baseSeverity computed from score (not in JSON)."""
    metrics = _make_nvd_metric(
        version="4.0",
        key_suffix="V40",
        base_score=9.3,
        base_severity=None,  # v4.0 has no baseSeverity in cvssData
    )
    entries = parse_cvss(metrics)
    assert len(entries) == 1
    assert entries[0].version == "4.0"
    assert entries[0].base_score == 9.3
    assert entries[0].base_severity == "CRITICAL"
    assert entries[0].source == "nvd"
    assert entries[0].is_authoritative is True


def test_parse_cvss_v40_score_zero() -> None:
    """CVSS 4.0 with score 0.0 computes NONE severity."""
    metrics = _make_nvd_metric(
        version="4.0", key_suffix="V40", base_score=0.0, base_severity=None
    )
    entries = parse_cvss(metrics)
    assert entries[0].base_severity == "NONE"
    assert entries[0].base_score == 0.0


def test_parse_cvss_v31_reads_severity_from_cvss_data() -> None:
    """CVSS 3.1: baseSeverity read from cvssData (not computed)."""
    metrics = _make_nvd_metric(version="3.1", key_suffix="V31", base_score=9.8)
    entries = parse_cvss(metrics)
    assert len(entries) == 1
    assert entries[0].version == "3.1"
    assert entries[0].base_score == 9.8
    assert entries[0].base_severity == "CRITICAL"


def test_parse_cvss_v2_severity_at_metric_level() -> None:
    """CVSS 2.0: baseSeverity at metric level, not in cvssData."""
    metrics = _make_nvd_metric(
        version="2.0",
        key_suffix="V2",
        base_score=4.3,
        base_severity="MEDIUM",
        severity_at_metric_level=True,
    )
    entries = parse_cvss(metrics)
    assert entries[0].base_severity == "MEDIUM"


def test_parse_cvss_multi_version_ordering() -> None:
    """Multiple CVSS versions parsed in order: v4.0, v3.1, v2.0."""
    metrics: dict[str, list[dict[str, Any]]] = {}
    metrics.update(
        _make_nvd_metric(
            version="4.0", key_suffix="V40", base_score=9.3, base_severity=None
        )
    )
    metrics.update(_make_nvd_metric(version="3.1", key_suffix="V31", base_score=9.8))
    metrics.update(
        _make_nvd_metric(
            version="2.0",
            key_suffix="V2",
            base_score=7.5,
            base_severity="HIGH",
            severity_at_metric_level=True,
        )
    )
    entries = parse_cvss(metrics)
    assert len(entries) == 3
    assert entries[0].version == "4.0"
    assert entries[1].version == "3.1"
    assert entries[2].version == "2.0"


def test_parse_cvss_empty() -> None:
    """Empty metrics returns empty list."""
    assert parse_cvss({}) == []


def test_parse_cvss_v40_extracts_components() -> None:
    """v4.0 parser reads vulnConfidentialityImpact etc. from cvssData."""
    metrics = {
        "cvssMetricV40": [
            {
                "source": "nvd@nist.gov",
                "type": "Primary",
                "cvssData": {
                    "version": "4.0",
                    "vectorString": "CVSS:4.0/AV:L/AC:L/AT:N/PR:L/UI:N/VC:H/VI:H/VA:H/SC:N/SI:N/SA:N",
                    "baseScore": 8.5,
                    "baseSeverity": "HIGH",
                    "attackVector": "LOCAL",
                    "attackComplexity": "LOW",
                    "attackRequirements": "NONE",
                    "privilegesRequired": "LOW",
                    "userInteraction": "NONE",
                    "vulnConfidentialityImpact": "HIGH",
                    "vulnIntegrityImpact": "HIGH",
                    "vulnAvailabilityImpact": "HIGH",
                    "subConfidentialityImpact": "NONE",
                    "subIntegrityImpact": "NONE",
                    "subAvailabilityImpact": "NONE",
                },
            }
        ],
    }
    entries = parse_cvss(metrics)
    assert len(entries) == 1
    e = entries[0]
    assert e.version == "4.0"
    assert e.attack_vector == "LOCAL"
    assert e.attack_requirements == "NONE"
    assert e.vuln_confidentiality_impact == "HIGH"
    assert e.sub_confidentiality_impact == "NONE"


def test_parse_cvss_v40_supplemental_metrics() -> None:
    """v4.0 parser reads supplemental metrics from cvssData."""
    metrics = {
        "cvssMetricV40": [
            {
                "source": "psirt@vendor.com",
                "type": "Secondary",
                "cvssData": {
                    "version": "4.0",
                    "vectorString": "CVSS:4.0/AV:N/AC:L/AT:N/PR:N/UI:N/VC:H/VI:H/VA:H/SC:N/SI:N/SA:N",
                    "baseScore": 9.3,
                    "baseSeverity": "CRITICAL",
                    "exploitMaturity": "ATTACKED",
                    "Automatable": "YES",
                    "Recovery": "AUTOMATIC",
                    "Safety": "PRESENT",
                    "providerUrgency": "RED",
                    "vulnerabilityResponseEffort": "LOW",
                    "valueDensity": "CONCENTRATED",
                },
            }
        ],
    }
    entries = parse_cvss(metrics)
    e = entries[0]
    assert e.exploit_maturity == "ATTACKED"
    assert e.automatable == "YES"
    assert e.recovery == "AUTOMATIC"
    assert e.safety == "PRESENT"
    assert e.provider_urgency == "RED"
    assert e.vulnerability_response_effort == "LOW"
    assert e.value_density == "CONCENTRATED"


def test_parse_cvss_v31_extracts_components() -> None:
    """v3.1 parser reads attackVector, scope, etc. from cvssData."""
    metrics = {
        "cvssMetricV31": [
            {
                "source": "nvd@nist.gov",
                "type": "Primary",
                "cvssData": {
                    "version": "3.1",
                    "vectorString": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
                    "baseScore": 9.8,
                    "baseSeverity": "CRITICAL",
                    "attackVector": "NETWORK",
                    "attackComplexity": "LOW",
                    "privilegesRequired": "NONE",
                    "userInteraction": "NONE",
                    "scope": "UNCHANGED",
                    "confidentialityImpact": "HIGH",
                    "integrityImpact": "HIGH",
                    "availabilityImpact": "HIGH",
                },
                "exploitabilityScore": 3.9,
                "impactScore": 5.9,
            }
        ],
    }
    entries = parse_cvss(metrics)
    e = entries[0]
    assert e.attack_vector == "NETWORK"
    assert e.scope == "UNCHANGED"
    assert e.confidentiality_impact == "HIGH"
    assert e.exploitability_score == 3.9


def test_parse_cvss_v20_extracts_components() -> None:
    """v2.0 parser reads accessVector (not attackVector) from cvssData."""
    metrics = {
        "cvssMetricV2": [
            {
                "source": "nvd@nist.gov",
                "type": "Primary",
                "cvssData": {
                    "version": "2.0",
                    "vectorString": "AV:N/AC:L/Au:N/C:C/I:C/A:C",
                    "baseScore": 10.0,
                    "accessVector": "NETWORK",
                    "accessComplexity": "LOW",
                    "confidentialityImpact": "COMPLETE",
                    "integrityImpact": "COMPLETE",
                    "availabilityImpact": "COMPLETE",
                },
                "baseSeverity": "HIGH",
                "exploitabilityScore": 10.0,
                "impactScore": 10.0,
            }
        ],
    }
    entries = parse_cvss(metrics)
    e = entries[0]
    assert e.attack_vector == "NETWORK"
    assert e.attack_complexity == "LOW"
    assert e.confidentiality_impact == "COMPLETE"
    assert e.privileges_required is None  # v2 doesn't have PR
    assert e.scope is None  # v2 doesn't have scope


# --- parse_cvss completeness guards ---
#
# These tests use cherry-picked NVD JSON with EVERY field populated.
# Every parseable field is asserted — if a parser line is accidentally
# deleted or a new schema field isn't wired up, these tests fail.
#
# Each test also includes a schema sync guard: it compares the set of
# asserted fields against the dataclass fields, so adding a field to
# CVSSv3Entry/CVSSv4Entry without updating the test is a failure.

# Fields that parse_cvss never populates (not in NVD data).
_UNPOPULATED_FIELDS = frozenset({"source_url", "published_at", "updated_at"})


class TestCVSSv40Completeness:
    """Completeness guard: parse_cvss extracts ALL v4.0 fields from NVD JSON."""

    # Cherry-picked NVD v4.0 metric with every field populated.
    # Based on real NVD API shape (CVE-2025-2502 style).
    FULL_METRIC: dict[str, Any] = {
        "source": "nvd@nist.gov",
        "type": "Primary",
        "cvssData": {
            "version": "4.0",
            "vectorString": "CVSS:4.0/AV:L/AC:L/AT:N/PR:L/UI:N/VC:H/VI:H/VA:H/SC:N/SI:N/SA:N",
            "baseScore": 8.5,
            "baseSeverity": "HIGH",
            # Base metrics
            "attackVector": "LOCAL",
            "attackComplexity": "LOW",
            "attackRequirements": "NONE",
            "privilegesRequired": "LOW",
            "userInteraction": "NONE",
            # Vulnerable system impacts
            "vulnConfidentialityImpact": "HIGH",
            "vulnIntegrityImpact": "HIGH",
            "vulnAvailabilityImpact": "HIGH",
            # Subsequent system impacts
            "subConfidentialityImpact": "NONE",
            "subIntegrityImpact": "NONE",
            "subAvailabilityImpact": "NONE",
            # Supplemental metrics (NB: NVD uses inconsistent casing)
            "exploitMaturity": "ATTACKED",
            "Automatable": "YES",
            "Recovery": "AUTOMATIC",
            "Safety": "PRESENT",
            "providerUrgency": "RED",
            "vulnerabilityResponseEffort": "LOW",
            "valueDensity": "CONCENTRATED",
        },
        "exploitabilityScore": 2.0,
        "impactScore": 3.6,
    }

    def test_all_fields_populated(self) -> None:
        """v4.0: every parseable CVSSv4Entry field extracted from NVD JSON."""
        entries = parse_cvss({"cvssMetricV40": [self.FULL_METRIC]})
        assert len(entries) == 1
        e = entries[0]
        assert isinstance(e, CVSSv4Entry)

        # Core fields
        assert e.version == "4.0"
        assert (
            e.vector_string
            == "CVSS:4.0/AV:L/AC:L/AT:N/PR:L/UI:N/VC:H/VI:H/VA:H/SC:N/SI:N/SA:N"
        )
        assert e.base_score == 8.5
        assert e.base_severity == "HIGH"
        assert e.source == "nvd"
        assert e.is_authoritative is True

        # Base metrics
        assert e.attack_vector == "LOCAL"
        assert e.attack_complexity == "LOW"
        assert e.attack_requirements == "NONE"
        assert e.privileges_required == "LOW"
        assert e.user_interaction == "NONE"

        # Vulnerable system impacts
        assert e.vuln_confidentiality_impact == "HIGH"
        assert e.vuln_integrity_impact == "HIGH"
        assert e.vuln_availability_impact == "HIGH"

        # Subsequent system impacts
        assert e.sub_confidentiality_impact == "NONE"
        assert e.sub_integrity_impact == "NONE"
        assert e.sub_availability_impact == "NONE"

        # Supplemental metrics
        assert e.exploit_maturity == "ATTACKED"
        assert e.automatable == "YES"
        assert e.recovery == "AUTOMATIC"
        assert e.safety == "PRESENT"
        assert e.provider_urgency == "RED"
        assert e.vulnerability_response_effort == "LOW"
        assert e.value_density == "CONCENTRATED"

        # Sub-scores (from metric wrapper, not cvssData)
        assert e.exploitability_score == 2.0
        assert e.impact_score == 3.6

        # Not in NVD data — must be None
        assert e.source_url is None
        assert e.published_at is None
        assert e.updated_at is None

    def test_schema_sync(self) -> None:
        """Guard: fails if CVSSv4Entry gains a field not covered by the completeness test."""
        from dataclasses import fields as dc_fields

        schema_fields = {f.name for f in dc_fields(CVSSv4Entry)}
        # Fields asserted in test_all_fields_populated (update when adding fields)
        tested_fields = {
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
            "metadata",
        }
        untested = schema_fields - tested_fields
        assert (
            not untested
        ), f"CVSSv4Entry has fields not covered by completeness test: {untested}"


class TestCVSSv31Completeness:
    """Completeness guard: parse_cvss extracts ALL v3.1 fields from NVD JSON."""

    FULL_METRIC: dict[str, Any] = {
        "source": "nvd@nist.gov",
        "type": "Primary",
        "cvssData": {
            "version": "3.1",
            "vectorString": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
            "baseScore": 9.8,
            "baseSeverity": "CRITICAL",
            "attackVector": "NETWORK",
            "attackComplexity": "LOW",
            "privilegesRequired": "NONE",
            "userInteraction": "NONE",
            "scope": "UNCHANGED",
            "confidentialityImpact": "HIGH",
            "integrityImpact": "HIGH",
            "availabilityImpact": "HIGH",
        },
        "exploitabilityScore": 3.9,
        "impactScore": 5.9,
    }

    def test_all_fields_populated(self) -> None:
        """v3.1: every parseable CVSSv3Entry field extracted from NVD JSON."""
        entries = parse_cvss({"cvssMetricV31": [self.FULL_METRIC]})
        assert len(entries) == 1
        e = entries[0]
        assert isinstance(e, CVSSv3Entry)

        # Core fields
        assert e.version == "3.1"
        assert e.vector_string == "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"
        assert e.base_score == 9.8
        assert e.base_severity == "CRITICAL"
        assert e.source == "nvd"
        assert e.is_authoritative is True

        # Component fields
        assert e.attack_vector == "NETWORK"
        assert e.attack_complexity == "LOW"
        assert e.privileges_required == "NONE"
        assert e.user_interaction == "NONE"
        assert e.scope == "UNCHANGED"
        assert e.confidentiality_impact == "HIGH"
        assert e.integrity_impact == "HIGH"
        assert e.availability_impact == "HIGH"

        # Sub-scores
        assert e.exploitability_score == 3.9
        assert e.impact_score == 5.9

        # Not in NVD data
        assert e.source_url is None
        assert e.published_at is None
        assert e.updated_at is None

    def test_schema_sync(self) -> None:
        """Guard: fails if CVSSv3Entry gains a field not covered by the completeness test."""
        from dataclasses import fields as dc_fields

        schema_fields = {f.name for f in dc_fields(CVSSv3Entry)}
        tested_fields = {
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
            "metadata",
        }
        untested = schema_fields - tested_fields
        assert (
            not untested
        ), f"CVSSv3Entry has fields not covered by completeness test: {untested}"


class TestCVSSv30Completeness:
    """Completeness guard: parse_cvss extracts ALL v3.0 fields from NVD JSON."""

    FULL_METRIC: dict[str, Any] = {
        "source": "security@vendor.com",
        "type": "Secondary",
        "cvssData": {
            "version": "3.0",
            "vectorString": "CVSS:3.0/AV:A/AC:H/PR:L/UI:R/S:C/C:L/I:L/A:L",
            "baseScore": 5.5,
            "baseSeverity": "MEDIUM",
            "attackVector": "ADJACENT_NETWORK",
            "attackComplexity": "HIGH",
            "privilegesRequired": "LOW",
            "userInteraction": "REQUIRED",
            "scope": "CHANGED",
            "confidentialityImpact": "LOW",
            "integrityImpact": "LOW",
            "availabilityImpact": "LOW",
        },
        "exploitabilityScore": 1.3,
        "impactScore": 3.7,
    }

    def test_all_fields_populated(self) -> None:
        """v3.0: every parseable CVSSv3Entry field extracted from NVD JSON."""
        entries = parse_cvss({"cvssMetricV30": [self.FULL_METRIC]})
        assert len(entries) == 1
        e = entries[0]
        assert isinstance(e, CVSSv3Entry)

        # Core fields
        assert e.version == "3.0"
        assert e.vector_string == "CVSS:3.0/AV:A/AC:H/PR:L/UI:R/S:C/C:L/I:L/A:L"
        assert e.base_score == 5.5
        assert e.base_severity == "MEDIUM"
        assert e.source == "vendor"
        assert e.is_authoritative is False

        # Component fields
        assert e.attack_vector == "ADJACENT_NETWORK"
        assert e.attack_complexity == "HIGH"
        assert e.privileges_required == "LOW"
        assert e.user_interaction == "REQUIRED"
        assert e.scope == "CHANGED"
        assert e.confidentiality_impact == "LOW"
        assert e.integrity_impact == "LOW"
        assert e.availability_impact == "LOW"

        # Sub-scores
        assert e.exploitability_score == 1.3
        assert e.impact_score == 3.7

        # Not in NVD data
        assert e.source_url is None
        assert e.published_at is None
        assert e.updated_at is None


class TestCVSSv2Completeness:
    """Completeness guard: parse_cvss extracts ALL v2.0 fields from NVD JSON."""

    FULL_METRIC: dict[str, Any] = {
        "source": "nvd@nist.gov",
        "type": "Primary",
        "cvssData": {
            "version": "2.0",
            "vectorString": "AV:N/AC:L/Au:N/C:C/I:C/A:C",
            "baseScore": 10.0,
            # v2 uses accessVector/accessComplexity (not attackVector)
            "accessVector": "NETWORK",
            "accessComplexity": "LOW",
            "authentication": "NONE",
            "confidentialityImpact": "COMPLETE",
            "integrityImpact": "COMPLETE",
            "availabilityImpact": "COMPLETE",
        },
        # v2: baseSeverity at metric level, not in cvssData
        "baseSeverity": "HIGH",
        "exploitabilityScore": 10.0,
        "impactScore": 10.0,
    }

    def test_all_fields_populated(self) -> None:
        """v2.0: every parseable field extracted, v3-only fields are None."""
        entries = parse_cvss({"cvssMetricV2": [self.FULL_METRIC]})
        assert len(entries) == 1
        e = entries[0]
        assert isinstance(e, CVSSv3Entry)  # v2 stored as CVSSv3Entry

        # Core fields
        assert e.version == "2.0"
        assert e.vector_string == "AV:N/AC:L/Au:N/C:C/I:C/A:C"
        assert e.base_score == 10.0
        assert e.base_severity == "HIGH"  # from metric level
        assert e.source == "nvd"
        assert e.is_authoritative is True

        # v2 field mapping: accessVector → attack_vector, accessComplexity → attack_complexity
        assert e.attack_vector == "NETWORK"
        assert e.attack_complexity == "LOW"

        # v2 CIA impacts
        assert e.confidentiality_impact == "COMPLETE"
        assert e.integrity_impact == "COMPLETE"
        assert e.availability_impact == "COMPLETE"

        # v3-only fields — must be None for v2
        assert e.privileges_required is None
        assert e.user_interaction is None
        assert e.scope is None

        # Sub-scores
        assert e.exploitability_score == 10.0
        assert e.impact_score == 10.0

        # Not in NVD data
        assert e.source_url is None
        assert e.published_at is None
        assert e.updated_at is None

    def test_schema_sync(self) -> None:
        """Guard: fails if CVSSv3Entry gains a field not covered by the v2 completeness test."""
        from dataclasses import fields as dc_fields

        schema_fields = {f.name for f in dc_fields(CVSSv3Entry)}
        # All fields must be accounted for (either populated or explicitly None)
        tested_fields = {
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
            "metadata",
        }
        untested = schema_fields - tested_fields
        assert (
            not untested
        ), f"CVSSv3Entry has fields not covered by v2 completeness test: {untested}"


# --- parse_cwes ---


def test_parse_cwes_basic() -> None:
    """Parse single CWE from weakness."""
    weaknesses = [{"description": [{"lang": "en", "value": "CWE-94"}]}]
    entries = parse_cwes(weaknesses)
    assert len(entries) == 1
    assert entries[0].id == "CWE-94"


def test_parse_cwes_skips_non_cwe() -> None:
    """Non-CWE values like NVD-CWE-noinfo are skipped."""
    weaknesses = [{"description": [{"lang": "en", "value": "NVD-CWE-noinfo"}]}]
    assert parse_cwes(weaknesses) == []


def test_parse_cwes_empty() -> None:
    """Empty weaknesses returns empty list."""
    assert parse_cwes([]) == []


# --- parse_cpes ---


def test_parse_cpes_basic() -> None:
    """Parse CPE with fix version from versionEndExcluding."""
    configurations = [
        {
            "nodes": [
                {
                    "cpeMatch": [
                        {
                            "criteria": "cpe:2.3:a:example:product:1.0:*:*:*:*:*:*:*",
                            "versionEndExcluding": "2.0.0",
                        }
                    ]
                }
            ]
        }
    ]
    entries = parse_cpes(configurations)
    assert len(entries) == 1
    assert entries[0].vendor == "example"
    assert entries[0].product == "product"
    assert entries[0].fix_version == "2.0.0"


def test_parse_cpes_skips_invalid_cpe_string() -> None:
    """Invalid CPE strings are silently skipped."""
    configurations = [
        {
            "nodes": [
                {
                    "cpeMatch": [
                        {"criteria": "cpe:2.3:INVALID"},
                        {"criteria": "cpe:2.3:a:valid:product:1.0:*:*:*:*:*:*:*"},
                    ]
                }
            ]
        }
    ]
    entries = parse_cpes(configurations)
    assert len(entries) == 1
    assert entries[0].vendor == "valid"


def test_parse_cpes_empty() -> None:
    """Empty configurations returns empty list."""
    assert parse_cpes([]) == []


# --- parse_references ---


def test_parse_references_basic() -> None:
    """Parse reference with tags."""
    refs = [
        {
            "url": "https://example.com/advisory",
            "source": "nvd@nist.gov",
            "tags": ["Vendor Advisory"],
        }
    ]
    entries = parse_references(refs)
    assert len(entries) == 1
    assert entries[0].url == "https://example.com/advisory"
    assert entries[0].tags == ["Vendor Advisory"]


def test_parse_references_no_tags() -> None:
    """Reference without tags gets empty list."""
    refs = [{"url": "https://example.com/advisory", "source": "nvd@nist.gov"}]
    entries = parse_references(refs)
    assert entries[0].tags == []


def test_parse_references_non_string_source_and_non_list_tags() -> None:
    """Non-string source becomes None, non-list tags becomes empty list."""
    refs = [
        {
            "url": "https://example.com/advisory",
            "source": 12345,
            "tags": "not-a-list",
        }
    ]
    entries = parse_references(refs)
    assert len(entries) == 1
    assert entries[0].source is None
    assert entries[0].tags == []


def test_parse_references_empty() -> None:
    """Empty references returns empty list."""
    assert parse_references([]) == []


# --- populate_columns ---


def test_populate_columns_returns_dict_of_arrays() -> None:
    """populate_columns converts vulnerability dicts to columnar arrays."""
    import json
    from pathlib import Path

    from vulnflow.utils.nvd import populate_columns

    sample = json.loads(Path("tests/fixtures/nvd_sample.json").read_text())
    vulns = sample["vulnerabilities"]

    arrays = populate_columns(vulns)

    assert isinstance(arrays, dict)
    assert "cve_id" in arrays
    assert arrays["cve_id"][0] == "CVE-2024-0001"
    assert len(arrays["cve_id"]) == 5


def test_populate_columns_empty_list() -> None:
    """An empty feed keeps the schema, so downstream transforms find their columns."""
    from vulnflow.utils.nvd import populate_columns

    arrays = populate_columns([])
    assert len(arrays["cve_id"]) == 0
