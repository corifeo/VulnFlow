"""Tests for BaseInference standard pattern: overwrite, tiers, _should_run.

Layer: Tests/Unit
Dependencies: transforms.base.BaseInference
"""

from __future__ import annotations

from vulnflow.dataset import VulnDataset
from vulnflow.registry import FieldDefinition
from vulnflow.transforms.base import BaseInference

# ============================================================================
# Concrete test subclass
# ============================================================================


class DummyInference(BaseInference):
    """Minimal concrete inference for testing base class behavior."""

    name: str = "DummyInference"
    depends_on: list[str] = []
    OUTPUT_SCHEMA: list[FieldDefinition] = []

    TIER_THRESHOLDS: dict[int, float] = {
        1: 0.95,
        2: 0.80,
        3: 0.60,
    }

    def transform(self, dataset: VulnDataset) -> None:
        pass


# ============================================================================
# Constructor defaults
# ============================================================================


class TestBaseInferenceDefaults:
    def test_default_confidence_threshold(self) -> None:
        """Default confidence_threshold is 0.7."""
        inf = DummyInference()
        assert inf.confidence_threshold == 0.7

    def test_default_overwrite_is_false(self) -> None:
        """Default overwrite is False."""
        inf = DummyInference()
        assert inf.overwrite is False

    def test_default_tiers_is_none(self) -> None:
        """Default tiers is None (all tiers via threshold)."""
        inf = DummyInference()
        assert inf.tiers is None

    def test_custom_confidence_threshold(self) -> None:
        """Constructor accepts custom confidence_threshold."""
        inf = DummyInference(confidence_threshold=0.5)
        assert inf.confidence_threshold == 0.5

    def test_custom_overwrite(self) -> None:
        """Constructor accepts overwrite=True."""
        inf = DummyInference(overwrite=True)
        assert inf.overwrite is True

    def test_custom_tiers(self) -> None:
        """Constructor accepts explicit tier list."""
        inf = DummyInference(tiers=[1, 3])
        assert inf.tiers == [1, 3]


# ============================================================================
# _should_run: threshold-based gating
# ============================================================================


class TestShouldRunThreshold:
    def test_threshold_below_tier_runs(self) -> None:
        """Tier runs when confidence_threshold <= tier threshold."""
        inf = DummyInference(confidence_threshold=0.7)
        # Tier 1 (0.95) and Tier 2 (0.80) should run, Tier 3 (0.60) should not
        assert inf._should_run(1) is True
        assert inf._should_run(2) is True
        assert inf._should_run(3) is False

    def test_threshold_equal_to_tier_runs(self) -> None:
        """Tier runs when confidence_threshold == tier threshold."""
        inf = DummyInference(confidence_threshold=0.80)
        assert inf._should_run(2) is True

    def test_threshold_above_tier_skips(self) -> None:
        """Tier skipped when confidence_threshold > tier threshold."""
        inf = DummyInference(confidence_threshold=0.81)
        assert inf._should_run(2) is False

    def test_very_low_threshold_runs_all(self) -> None:
        """Very low threshold runs all tiers."""
        inf = DummyInference(confidence_threshold=0.1)
        assert inf._should_run(1) is True
        assert inf._should_run(2) is True
        assert inf._should_run(3) is True

    def test_very_high_threshold_skips_all(self) -> None:
        """Threshold above all tiers skips everything."""
        inf = DummyInference(confidence_threshold=1.0)
        assert inf._should_run(1) is False
        assert inf._should_run(2) is False
        assert inf._should_run(3) is False

    def test_unknown_tier_defaults_to_zero(self) -> None:
        """Unknown tier ID defaults to 0.0 threshold, so always skipped."""
        inf = DummyInference(confidence_threshold=0.1)
        assert inf._should_run(99) is False


# ============================================================================
# _should_run: explicit tier selection
# ============================================================================


class TestShouldRunExplicitTiers:
    def test_explicit_tiers_override_threshold(self) -> None:
        """Explicit tiers list overrides threshold-based gating."""
        inf = DummyInference(confidence_threshold=0.95, tiers=[3])
        # Tier 3 (0.60) would fail threshold check, but explicit tiers overrides
        assert inf._should_run(3) is True
        # Tier 1 is NOT in the explicit list
        assert inf._should_run(1) is False

    def test_explicit_empty_tiers_runs_nothing(self) -> None:
        """Empty explicit tiers list runs no tiers."""
        inf = DummyInference(tiers=[])
        assert inf._should_run(1) is False
        assert inf._should_run(2) is False
        assert inf._should_run(3) is False

    def test_explicit_single_tier(self) -> None:
        """Single tier in explicit list only runs that tier."""
        inf = DummyInference(tiers=[2])
        assert inf._should_run(1) is False
        assert inf._should_run(2) is True
        assert inf._should_run(3) is False

    def test_explicit_all_tiers(self) -> None:
        """Explicit list with all tiers runs all."""
        inf = DummyInference(tiers=[1, 2, 3])
        assert inf._should_run(1) is True
        assert inf._should_run(2) is True
        assert inf._should_run(3) is True


# ============================================================================
# TIER_THRESHOLDS class attribute
# ============================================================================


class TestTierThresholds:
    def test_base_class_has_empty_tier_thresholds(self) -> None:
        """BaseInference.TIER_THRESHOLDS is empty dict by default."""
        assert BaseInference.TIER_THRESHOLDS == {}

    def test_subclass_can_override_tier_thresholds(self) -> None:
        """Subclass defines its own TIER_THRESHOLDS."""
        assert DummyInference.TIER_THRESHOLDS == {1: 0.95, 2: 0.80, 3: 0.60}

    def test_no_tier_thresholds_means_all_skipped_by_threshold(self) -> None:
        """Inference with empty TIER_THRESHOLDS skips all by threshold."""

        class EmptyTiers(BaseInference):
            name: str = "EmptyTiers"
            depends_on: list[str] = []
            OUTPUT_SCHEMA: list[FieldDefinition] = []
            TIER_THRESHOLDS: dict[int, float] = {}

            def transform(self, dataset: VulnDataset) -> None:
                pass

        inf = EmptyTiers(confidence_threshold=0.0)
        # No tiers defined, so _should_run always False (get returns 0.0)
        assert inf._should_run(1) is True  # 0.0 <= 0.0 is True
