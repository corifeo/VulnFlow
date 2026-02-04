"""
CVD State Transform - Compute all CVD analytics from timestamps

Provides:
- CVDStateTransform: Computes all 19 CVD fields from timestamps

Layer: Transform/Computed
Dependencies: EventInference (provides CVD timestamps)

Fields computed (19 total):
  Core (2):        cvd_state, state_bits
  Dimensions (2):  fix_path, threat_state
  Pairwise (1):    pair_mask
  Desiderata (2):  desiderata_mask, desiderata_count
  Zero-Day (3):    is_zero_day_exploit, is_zero_day_attack, is_uncoordinated
  Exposure (3):    is_premature_disclosure, is_exploit_before_fix, is_attack_before_fix
  Time Deltas (6): fix_lag_days, disclosure_lag_days, deployment_lag_days,
                   exploit_lag_days, attack_lag_days, public_to_attack_days

Based on SEI/CMU 2021 paper and vulnstate/constants.py definitions.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from vulnflow.dataset import VulnDataset

from vulnflow.registry import FieldDefinition, register_transform
from vulnflow.transforms.base import BaseComputed


def _safe_compare(ts1: datetime | None, ts2: datetime | None) -> bool | None:
    """Compare two timestamps safely, handling timezone mismatches.

    Returns True if ts1 < ts2, False if ts1 >= ts2, None if either is None.
    Normalizes timezone-naive datetimes to UTC for comparison.
    """
    if ts1 is None or ts2 is None:
        return None

    # Normalize to UTC if needed
    if ts1.tzinfo is None:
        ts1 = ts1.replace(tzinfo=UTC)
    if ts2.tzinfo is None:
        ts2 = ts2.replace(tzinfo=UTC)

    return ts1 < ts2


def _safe_delta(ts1: datetime | None, ts2: datetime | None) -> float:
    """Compute time delta in days between two timestamps.

    Returns NaN if either timestamp is None.
    Normalizes timezone-naive datetimes to UTC.
    """
    if ts1 is None or ts2 is None:
        return float("nan")

    # Normalize to UTC if needed
    if ts1.tzinfo is None:
        ts1 = ts1.replace(tzinfo=UTC)
    if ts2.tzinfo is None:
        ts2 = ts2.replace(tzinfo=UTC)

    return (ts2 - ts1).total_seconds() / 86400.0


# =============================================================================
# Constants from CVD model (matching vulnstate/constants.py)
# =============================================================================

# Fix Path values (VFD dimension, bits 0-2)
FIX_PATH_NO_AWARENESS = 0b000  # vfd
FIX_PATH_VENDOR_AWARE = 0b001  # Vfd
FIX_PATH_FIX_READY = 0b011  # VFd
FIX_PATH_REMEDIATED = 0b111  # VFD

# Threat State values (PXA dimension, bits 3-5)
THREAT_LATENT = 0b000  # pxa
THREAT_DISCLOSED = 0b001  # Pxa
THREAT_WEAPONIZED = 0b011  # PXa
THREAT_TARGETED = 0b100  # pxA
THREAT_UNDER_ATTACK = 0b101  # PxA
THREAT_ACTIVE = 0b111  # PXA

# Pair mask bit positions (15 pairs)
# Bit 0: V<F, Bit 1: V<D, Bit 2: V<P, Bit 3: V<X, Bit 4: V<A
# Bit 5: F<D, Bit 6: F<P, Bit 7: F<X, Bit 8: F<A
# Bit 9: D<P, Bit 10: D<X, Bit 11: D<A
# Bit 12: P<X, Bit 13: P<A, Bit 14: X<A

# Desiderata bit positions (12 desired orderings)
# Maps to pair_mask bits for the 12 desiderata
DESIDERATA_BITS = [2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14]
# D1=V<P(2), D2=V<X(3), D3=V<A(4), D4=F<P(6), D5=F<X(7), D6=F<A(8)
# D7=D<P(9), D8=D<X(10), D9=D<A(11), D10=P<X(12), D11=P<A(13), D12=X<A(14)

# Anti-desiderata (violations) - computed from pair_mask
# U1: P<V (bit 2 NOT set when both exist)
# U2: X<V (bit 3 NOT set) - zero-day exploit
# U3: A<V (bit 4 NOT set) - zero-day attack
# U4: P<F (bit 6 NOT set) - premature disclosure
# U5: X<F (bit 7 NOT set) - exploit before fix
# U6: A<F (bit 8 NOT set) - attack before fix


@register_transform(default=True)
class CVDStateTransform(BaseComputed):
    """Compute all 19 CVD analytics fields from timestamps.

    Requires EventInference to run first (provides CVD timestamp columns).

    This transform consolidates state computation, desiderata analysis,
    and time delta calculation into a single pass for efficiency.
    """

    name: str = "CVDStateTransform"
    depends_on: list[str] = ["EventInference"]

    SCHEMA_VERSION: str = "2.0.0"
    OUTPUT_SCHEMA: list[FieldDefinition] = [
        # Core (2)
        FieldDefinition(
            "cvd_state",
            "str",
            "6-character CVD state (e.g., 'VFdPxa')",
            "CVDStateTransform",
        ),
        FieldDefinition(
            "state_bits", "uint8", "6-bit state encoding (0-63)", "CVDStateTransform"
        ),
        # Dimensions (2)
        FieldDefinition(
            "fix_path",
            "uint8",
            "Fix path: 0=vfd, 1=Vfd, 3=VFd, 7=VFD",
            "CVDStateTransform",
        ),
        FieldDefinition(
            "threat_state",
            "uint8",
            "Threat state: 0=pxa, 1=Pxa, 3=PXa, 4=pxA, 5=PxA, 7=PXA",
            "CVDStateTransform",
        ),
        # Pairwise (1)
        FieldDefinition(
            "pair_mask",
            "uint16",
            "15-bit mask of pairwise orderings",
            "CVDStateTransform",
        ),
        # Desiderata (2)
        FieldDefinition(
            "desiderata_mask",
            "uint16",
            "12-bit mask of satisfied desiderata",
            "CVDStateTransform",
        ),
        FieldDefinition(
            "desiderata_count",
            "uint8",
            "Count of satisfied desiderata (0-12)",
            "CVDStateTransform",
        ),
        # Zero-Day flags (3)
        FieldDefinition(
            "is_zero_day_exploit",
            "bool",
            "X occurred before V (U2)",
            "CVDStateTransform",
        ),
        FieldDefinition(
            "is_zero_day_attack",
            "bool",
            "A occurred before V (U3)",
            "CVDStateTransform",
        ),
        FieldDefinition(
            "is_uncoordinated", "bool", "P occurred before V (U1)", "CVDStateTransform"
        ),
        # Exposure flags (3)
        FieldDefinition(
            "is_premature_disclosure",
            "bool",
            "P occurred before F (U4)",
            "CVDStateTransform",
        ),
        FieldDefinition(
            "is_exploit_before_fix",
            "bool",
            "X occurred before F (U5)",
            "CVDStateTransform",
        ),
        FieldDefinition(
            "is_attack_before_fix",
            "bool",
            "A occurred before F (U6)",
            "CVDStateTransform",
        ),
        # Time deltas (6)
        FieldDefinition(
            "fix_lag_days", "float32", "Days from V to F", "CVDStateTransform"
        ),
        FieldDefinition(
            "disclosure_lag_days", "float32", "Days from V to P", "CVDStateTransform"
        ),
        FieldDefinition(
            "deployment_lag_days", "float32", "Days from F to D", "CVDStateTransform"
        ),
        FieldDefinition(
            "exploit_lag_days", "float32", "Days from P to X", "CVDStateTransform"
        ),
        FieldDefinition(
            "attack_lag_days", "float32", "Days from X to A", "CVDStateTransform"
        ),
        FieldDefinition(
            "public_to_attack_days", "float32", "Days from P to A", "CVDStateTransform"
        ),
    ]

    def transform(self, dataset: VulnDataset) -> None:
        """Compute all 19 CVD fields from timestamps.

        Args:
            dataset: VulnDataset with CVD timestamp columns
        """
        n = len(dataset)

        # Get CVD timestamp columns
        v_ts = dataset.get_column_or_none("v_timestamp")
        f_ts = dataset.get_column_or_none("f_timestamp")
        d_ts = dataset.get_column_or_none("d_timestamp")
        p_ts = dataset.get_column_or_none("p_timestamp")
        x_ts = dataset.get_column_or_none("x_timestamp")
        a_ts = dataset.get_column_or_none("a_timestamp")

        # Pre-allocate all output arrays
        states: list[str] = []
        state_bits = np.zeros(n, dtype=np.uint8)
        fix_path = np.zeros(n, dtype=np.uint8)
        threat_state = np.zeros(n, dtype=np.uint8)
        pair_masks = np.zeros(n, dtype=np.uint16)
        desiderata_masks = np.zeros(n, dtype=np.uint16)
        desiderata_counts = np.zeros(n, dtype=np.uint8)

        # Boolean flags
        is_zero_day_exploit = np.zeros(n, dtype=bool)
        is_zero_day_attack = np.zeros(n, dtype=bool)
        is_uncoordinated = np.zeros(n, dtype=bool)
        is_premature_disclosure = np.zeros(n, dtype=bool)
        is_exploit_before_fix = np.zeros(n, dtype=bool)
        is_attack_before_fix = np.zeros(n, dtype=bool)

        # Time deltas
        fix_lag_days = np.full(n, np.nan, dtype=np.float32)
        disclosure_lag_days = np.full(n, np.nan, dtype=np.float32)
        deployment_lag_days = np.full(n, np.nan, dtype=np.float32)
        exploit_lag_days = np.full(n, np.nan, dtype=np.float32)
        attack_lag_days = np.full(n, np.nan, dtype=np.float32)
        public_to_attack_days = np.full(n, np.nan, dtype=np.float32)

        # Single pass through all records
        for i in range(n):
            # Get timestamps for this record
            v = v_ts[i] if v_ts is not None else None
            f = f_ts[i] if f_ts is not None else None
            d = d_ts[i] if d_ts is not None else None
            p = p_ts[i] if p_ts is not None else None
            x = x_ts[i] if x_ts is not None else None
            a = a_ts[i] if a_ts is not None else None

            # Event occurrence flags
            v_occurred = v is not None
            f_occurred = f is not None
            d_occurred = d is not None
            p_occurred = p is not None
            x_occurred = x is not None
            a_occurred = a is not None

            # === Core: cvd_state and state_bits ===
            state_str = (
                ("V" if v_occurred else "v")
                + ("F" if f_occurred else "f")
                + ("D" if d_occurred else "d")
                + ("P" if p_occurred else "p")
                + ("X" if x_occurred else "x")
                + ("A" if a_occurred else "a")
            )
            states.append(state_str)

            bits = (
                (1 if v_occurred else 0)
                | (2 if f_occurred else 0)
                | (4 if d_occurred else 0)
                | (8 if p_occurred else 0)
                | (16 if x_occurred else 0)
                | (32 if a_occurred else 0)
            )
            state_bits[i] = bits

            # === Dimensions: fix_path and threat_state ===
            vfd_bits = bits & 0b000111  # Lower 3 bits
            pxa_bits = (bits >> 3) & 0b000111  # Upper 3 bits
            fix_path[i] = vfd_bits
            threat_state[i] = pxa_bits

            # === Pairwise: pair_mask (15 bits) ===
            mask = 0
            if _safe_compare(v, f) is True:
                mask |= 1 << 0  # V<F
            if _safe_compare(v, d) is True:
                mask |= 1 << 1  # V<D
            if _safe_compare(v, p) is True:
                mask |= 1 << 2  # V<P
            if _safe_compare(v, x) is True:
                mask |= 1 << 3  # V<X
            if _safe_compare(v, a) is True:
                mask |= 1 << 4  # V<A
            if _safe_compare(f, d) is True:
                mask |= 1 << 5  # F<D
            if _safe_compare(f, p) is True:
                mask |= 1 << 6  # F<P
            if _safe_compare(f, x) is True:
                mask |= 1 << 7  # F<X
            if _safe_compare(f, a) is True:
                mask |= 1 << 8  # F<A
            if _safe_compare(d, p) is True:
                mask |= 1 << 9  # D<P
            if _safe_compare(d, x) is True:
                mask |= 1 << 10  # D<X
            if _safe_compare(d, a) is True:
                mask |= 1 << 11  # D<A
            if _safe_compare(p, x) is True:
                mask |= 1 << 12  # P<X
            if _safe_compare(p, a) is True:
                mask |= 1 << 13  # P<A
            if _safe_compare(x, a) is True:
                mask |= 1 << 14  # X<A
            pair_masks[i] = mask

            # === Desiderata: mask and count ===
            # Extract 12 desiderata bits from pair_mask
            d_mask = 0
            for bit_idx, pair_bit in enumerate(DESIDERATA_BITS):
                if mask & (1 << pair_bit):
                    d_mask |= 1 << bit_idx
            desiderata_masks[i] = d_mask
            desiderata_counts[i] = bin(d_mask).count("1")

            # === Zero-Day flags (anti-desiderata U1-U3) ===
            # These are TRUE when the bad ordering occurred
            if _safe_compare(p, v) is True:
                is_uncoordinated[i] = True  # U1: P before V
            if _safe_compare(x, v) is True:
                is_zero_day_exploit[i] = True  # U2: X before V
            if _safe_compare(a, v) is True:
                is_zero_day_attack[i] = True  # U3: A before V

            # === Exposure flags (anti-desiderata U4-U6) ===
            if _safe_compare(p, f) is True:
                is_premature_disclosure[i] = True  # U4: P before F
            if _safe_compare(x, f) is True:
                is_exploit_before_fix[i] = True  # U5: X before F
            if _safe_compare(a, f) is True:
                is_attack_before_fix[i] = True  # U6: A before F

            # === Time deltas ===
            fix_lag_days[i] = _safe_delta(v, f)
            disclosure_lag_days[i] = _safe_delta(v, p)
            deployment_lag_days[i] = _safe_delta(f, d)
            exploit_lag_days[i] = _safe_delta(p, x)
            attack_lag_days[i] = _safe_delta(x, a)
            public_to_attack_days[i] = _safe_delta(p, a)

        # Set all columns
        dataset.set_column("cvd_state", np.array(states, dtype=object))
        dataset.set_column("state_bits", state_bits)
        dataset.set_column("fix_path", fix_path)
        dataset.set_column("threat_state", threat_state)
        dataset.set_column("pair_mask", pair_masks)
        dataset.set_column("desiderata_mask", desiderata_masks)
        dataset.set_column("desiderata_count", desiderata_counts)
        dataset.set_column("is_zero_day_exploit", is_zero_day_exploit)
        dataset.set_column("is_zero_day_attack", is_zero_day_attack)
        dataset.set_column("is_uncoordinated", is_uncoordinated)
        dataset.set_column("is_premature_disclosure", is_premature_disclosure)
        dataset.set_column("is_exploit_before_fix", is_exploit_before_fix)
        dataset.set_column("is_attack_before_fix", is_attack_before_fix)
        dataset.set_column("fix_lag_days", fix_lag_days)
        dataset.set_column("disclosure_lag_days", disclosure_lag_days)
        dataset.set_column("deployment_lag_days", deployment_lag_days)
        dataset.set_column("exploit_lag_days", exploit_lag_days)
        dataset.set_column("attack_lag_days", attack_lag_days)
        dataset.set_column("public_to_attack_days", public_to_attack_days)
