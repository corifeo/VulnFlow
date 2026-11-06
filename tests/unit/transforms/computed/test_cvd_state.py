"""
CVD State Transform - Compute state from timestamps
Tests: State string computation, dependency on EventInference

Layer: Transform/Computed
Dependencies: EventInference
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from tests.conftest import make_dataset

from vulnflow.dataset import VulnDataset
from vulnflow.exceptions import TransformError
from vulnflow.transforms.computed.cvd_state import CVDStateTransform
from vulnflow.transforms.enrichers.kev import KEVEnricher
from vulnflow.transforms.flatten.kev import KEVFlattenTransform
from vulnflow.transforms.inferences.events import EventInference


def _row(**events: Any) -> VulnDataset:
    """Run CVDStateTransform on one CVE with the given v/f/d/p/x/a timestamps."""
    columns = {f"{e}_timestamp": [events.get(e)] for e in "vfdpxa"}
    ds = make_dataset({"cve_id": ["CVE-2024-0001"], **columns})
    CVDStateTransform().transform(ds)
    return ds


def _kev_chain() -> VulnDataset:
    return VulnDataset.from_nvd("tests/fixtures/nvd_sample.json").transform(
        [
            KEVEnricher(source="tests/fixtures/kev_sample.csv"),
            KEVFlattenTransform(),
            EventInference(),
            CVDStateTransform(),
        ]
    )


def _state_and_flags(ds: VulnDataset) -> tuple[list[Any], list[Any]]:
    return list(ds.get_column("cvd_state")), list(ds.get_column("is_zero_day_attack"))


def test_cvd_state_without_timestamps_raises() -> None:
    """No timestamp columns would report 'vfdpxa' for every CVE, which is junk."""
    ds = make_dataset({"cve_id": ["CVE-2024-0001"]})
    with pytest.raises(TransformError, match="EventInference"):
        CVDStateTransform().transform(ds)


def test_cvd_state_empty_dataset_without_timestamps_is_fine() -> None:
    """A feed with no new CVEs isn't an error."""
    ds = make_dataset({"cve_id": []})
    CVDStateTransform().transform(ds)
    assert len(ds.get_column("cvd_state")) == 0


def test_string_timestamps_are_compared() -> None:
    """ISO strings count as dates: JSON reloads and custom sources produce them."""
    ds = _row(a="2024-01-01T00:00:00+00:00", v=datetime(2024, 2, 1, tzinfo=UTC))
    assert ds.get_column("is_zero_day_attack")[0]


def test_kev_chain_populates_attack_analytics() -> None:
    """KEV dates reach the attack analytics through the full chain."""
    # CVE-2024-0001: published 2024-01-10T10:00Z, KEV 2024-01-11
    days = _kev_chain().get_column("public_to_attack_days")[0]
    assert days == pytest.approx(14 / 24, rel=1e-3)


def test_nat_is_not_an_event() -> None:
    """Null dates reloaded from Parquet don't count as events that happened."""
    nat = np.datetime64("NaT", "us")
    ds = _row(v=nat, f=nat, d=nat, p=np.datetime64("2024-01-01"), x=nat, a=nat)
    assert ds.get_column("cvd_state")[0] == "vfdPxa"


def test_malformed_timestamp_raises() -> None:
    """A bad date fails loudly instead of silently dropping the event."""
    with pytest.raises(ValueError):
        _row(a="not-a-date")


def test_parquet_roundtrip_keeps_state_and_flags(tmp_path: Path) -> None:
    ds = _kev_chain()
    ds.to_parquet(tmp_path / "x.parquet")
    again = VulnDataset.from_parquet(str(tmp_path / "x.parquet"))
    CVDStateTransform().transform(again)
    assert _state_and_flags(again) == _state_and_flags(ds)


def test_json_roundtrip_keeps_flags(tmp_path: Path) -> None:
    ds = _kev_chain()
    ds.to_json(tmp_path / "x.json")
    again = VulnDataset.from_json(tmp_path / "x.json")
    CVDStateTransform().transform(again)
    assert _state_and_flags(again) == _state_and_flags(ds)


def test_missing_v_is_not_a_zero_day() -> None:
    """No V means unknown: V is inferred, so absence isn't evidence of unawareness."""
    ds = _row(x=datetime(2024, 1, 1, tzinfo=UTC), a=datetime(2024, 1, 2, tzinfo=UTC))
    assert not ds.get_column("is_zero_day_exploit")[0]
    assert not ds.get_column("is_zero_day_attack")[0]


def test_same_day_kev_date_sorts_before_publication() -> None:
    """A KEV date means attacks were already seen by that day."""
    published = datetime(2024, 3, 14, 17, 15, tzinfo=UTC)
    ds = _row(v=published, p=published, a="2024-03-14T00:00:00+00:00")
    assert ds.get_column("is_zero_day_attack")[0]
    assert ds.get_column("public_to_attack_days")[0] < 0


def test_cvd_state_creates_state_column():
    """CVDStateTransform creates cvd_state column."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    # Run EventInference first (provides CVD timestamps)
    EventInference().transform(dataset)

    # Then run CVDStateTransform
    CVDStateTransform().transform(dataset)

    assert "cvd_state" in dataset.columns


def test_cvd_state_format():
    """CVD state is 6 characters (VFdPxa format)."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    EventInference().transform(dataset)
    CVDStateTransform().transform(dataset)

    states = dataset.get_column("cvd_state")
    for state in states:
        assert len(state) == 6
        # Each position should be uppercase (event occurred) or lowercase (not occurred)
        assert state[0] in "Vv"
        assert state[1] in "Ff"
        assert state[2] in "Dd"
        assert state[3] in "Pp"
        assert state[4] in "Xx"
        assert state[5] in "Aa"


def test_cvd_state_reflects_timestamps():
    """CVD state reflects which timestamps are set."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    # EventInference sets P from nvd_published_at
    EventInference().transform(dataset)
    CVDStateTransform().transform(dataset)

    states = dataset.get_column("cvd_state")
    # First record should have P (public disclosure) from nvd_published_at
    assert states[0][3] == "P"  # P is at position 3


def test_cvd_state_with_kev():
    """CVD state reflects A timestamp from KEV."""
    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")

    # Enrich with KEV, flatten, infer, compute state
    dataset.transform(
        [
            KEVEnricher(source="tests/fixtures/kev_sample.csv"),
            KEVFlattenTransform(),
            EventInference(),
            CVDStateTransform(),
        ]
    )

    states = dataset.get_column("cvd_state")
    # CVE-2024-0001 is in KEV, should have A
    assert states[0][5] == "A"  # A is at position 5
    # CVE-2024-0002 is NOT in KEV, should not have A
    assert states[1][5] == "a"


def test_cvd_state_registered():
    """CVDStateTransform is registered in transform registry."""
    from vulnflow.registry import get_transforms

    # Import to ensure registration
    from vulnflow.transforms.computed.cvd_state import CVDStateTransform  # noqa: F401

    transforms = get_transforms()
    transform_names = [t.__name__ for t in transforms]

    assert "CVDStateTransform" in transform_names


def test_cvd_state_adds_pair_mask():
    """CVDStateTransform adds pair_mask column."""
    from datetime import datetime

    import numpy as np

    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")
    n = len(dataset)

    # Add mock CVD timestamps
    v_ts = np.full(n, datetime(2024, 1, 1), dtype=object)
    f_ts = np.full(n, datetime(2024, 1, 15), dtype=object)
    p_ts = np.full(n, datetime(2024, 1, 10), dtype=object)

    dataset.set_column("v_timestamp", v_ts)
    dataset.set_column("f_timestamp", f_ts)
    dataset.set_column("d_timestamp", np.full(n, None, dtype=object))
    dataset.set_column("p_timestamp", p_ts)
    dataset.set_column("x_timestamp", np.full(n, None, dtype=object))
    dataset.set_column("a_timestamp", np.full(n, None, dtype=object))

    # Apply CVDStateTransform
    CVDStateTransform().transform(dataset)

    # Should have pair_mask column
    assert "pair_mask" in dataset.columns

    pair_mask = dataset.get_column("pair_mask")

    # V < F should set bit 0
    # V < P should set bit 2
    # Bit 0 = V<F (set), Bit 2 = V<P (set)
    assert pair_mask[0] & (1 << 0) != 0  # V<F
    assert pair_mask[0] & (1 << 2) != 0  # V<P


def test_cvd_state_adds_time_deltas():
    """CVDStateTransform adds time delta columns."""
    from datetime import datetime

    import numpy as np

    dataset = VulnDataset.from_nvd("tests/fixtures/nvd_sample.json")
    n = len(dataset)

    # V = day 1, F = day 16 (15 day lag)
    v_ts = np.full(n, datetime(2024, 1, 1), dtype=object)
    f_ts = np.full(n, datetime(2024, 1, 16), dtype=object)

    dataset.set_column("v_timestamp", v_ts)
    dataset.set_column("f_timestamp", f_ts)
    dataset.set_column("d_timestamp", np.full(n, None, dtype=object))
    dataset.set_column("p_timestamp", np.full(n, None, dtype=object))
    dataset.set_column("x_timestamp", np.full(n, None, dtype=object))
    dataset.set_column("a_timestamp", np.full(n, None, dtype=object))

    CVDStateTransform().transform(dataset)

    # Should have time delta columns
    assert "fix_lag_days" in dataset.columns
    assert "disclosure_lag_days" in dataset.columns
    assert "deployment_lag_days" in dataset.columns
    assert "exploit_lag_days" in dataset.columns
    assert "attack_lag_days" in dataset.columns
    assert "public_to_attack_days" in dataset.columns

    fix_lag_days = dataset.get_column("fix_lag_days")

    # V to F = 15 days
    assert fix_lag_days[0] == 15.0


def test_cvd_state_empty_dataset() -> None:
    """CVDStateTransform handles 0-row dataset without error."""
    ds = make_dataset(
        {
            "cve_id": [],
            "v_timestamp": [],
            "f_timestamp": [],
            "d_timestamp": [],
            "p_timestamp": [],
            "x_timestamp": [],
            "a_timestamp": [],
        }
    )
    CVDStateTransform().transform(ds)
    assert len(ds.get_column("cvd_state")) == 0
