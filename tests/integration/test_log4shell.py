"""
Log4Shell known answers - CVE-2021-44228 through the full CVD chain
Tests: Recorded NVD record + KEV entry give the expected state and flags

Layer: Integration Test
Dependencies: All layers
"""

from __future__ import annotations

from typing import Any

import pytest

from vulnflow import VulnPipeline
from vulnflow.extractors.nvd import NVDExtractor
from vulnflow.transforms.computed import CVDStateTransform
from vulnflow.transforms.enrichers import KEVEnricher
from vulnflow.transforms.flatten import KEVFlattenTransform
from vulnflow.transforms.inferences import EventInference


@pytest.fixture(scope="module")
def log4shell() -> dict[str, Any]:
    result = VulnPipeline(
        extractors=[NVDExtractor("tests/fixtures/nvd_log4shell.json")],
        transformers=[
            KEVEnricher(source="tests/fixtures/kev_log4shell.csv"),
            KEVFlattenTransform(),
            EventInference(),
            CVDStateTransform(),
        ],
    ).run(quiet=True, strict=True)
    assert result.dataset is not None
    return {c: result.dataset.get_column(c)[0] for c in result.dataset.columns}


def test_all_five_observable_events_are_present(log4shell: dict[str, Any]) -> None:
    assert log4shell["cvd_state"] == "VFdPXA"


def test_zero_day_flags_follow_the_dates(log4shell: dict[str, Any]) -> None:
    # V (Dec 11) comes from a URL date. Apache really knew in November, so this
    # pins what public data lets vulnflow infer, not what happened.
    assert log4shell["is_zero_day_exploit"]
    assert log4shell["is_zero_day_attack"]
    assert log4shell["is_uncoordinated"]


def test_exposure_flags_follow_the_dates(log4shell: dict[str, Any]) -> None:
    assert log4shell["is_premature_disclosure"]
    assert log4shell["is_exploit_before_fix"]
    assert log4shell["is_attack_before_fix"]


def test_kev_listing_precedes_nvd_publication(log4shell: dict[str, Any]) -> None:
    # KEV 2021-12-10 (midnight UTC), NVD published 2021-12-10T10:15:09Z
    assert log4shell["public_to_attack_days"] == pytest.approx(-10.25 / 24, rel=1e-2)


def test_no_desiderata_are_satisfied(log4shell: dict[str, Any]) -> None:
    assert log4shell["desiderata_count"] == 0
