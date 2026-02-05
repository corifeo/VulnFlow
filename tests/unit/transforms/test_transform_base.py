"""Tests for transform base class hierarchy."""

from __future__ import annotations

import pytest

from vulnflow.dataset import VulnDataset
from vulnflow.registry import FieldDefinition, Registry
from vulnflow.transforms.base import (
    BaseComputed,
    BaseEnricher,
    BaseInference,
    BaseTransform,
)


class TestBaseTransform:
    """Tests for BaseTransform root class."""

    def test_base_transform_is_abstract(self) -> None:
        """BaseTransform cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract"):
            BaseTransform()  # type: ignore[abstract]

    def test_base_transform_requires_name(self) -> None:
        """Subclass must define name attribute."""

        class BadTransform(BaseTransform):
            OUTPUT_SCHEMA: list[FieldDefinition] = []

            def transform(self, dataset: VulnDataset) -> None:
                pass

        with pytest.raises(TypeError):
            BadTransform()

    def test_base_transform_requires_output_schema(self) -> None:
        """Subclass must define OUTPUT_SCHEMA."""

        class BadTransform(BaseTransform):
            name = "test"

            def transform(self, dataset: VulnDataset) -> None:
                pass

        with pytest.raises(AttributeError):
            BadTransform()

    def test_base_transform_validates_schema_format(self) -> None:
        """OUTPUT_SCHEMA must be list[FieldDefinition]."""

        class BadTransform(BaseTransform):
            name = "test"
            OUTPUT_SCHEMA = [{"name": "foo"}]  # type: ignore[assignment]

            def transform(self, dataset: VulnDataset) -> None:
                pass

        with pytest.raises(TypeError, match="FieldDefinition"):
            BadTransform()

    def test_base_transform_auto_registers_schema(self) -> None:
        """BaseTransform auto-registers schema on init."""
        Registry.clear()

        class TestTransform(BaseTransform):
            name = "TestTransform"
            OUTPUT_SCHEMA = [
                FieldDefinition("test_field", "str", "Test", "TestTransform")
            ]
            depends_on: list[str] = []

            def transform(self, dataset: VulnDataset) -> None:
                pass

        TestTransform()

        columns = Registry.get_component_columns("TestTransform")
        assert len(columns) == 1
        assert columns[0].name == "test_field"

    def test_base_transform_requires_explicit_depends_on(self) -> None:
        """Transform with explicit depends_on succeeds."""

        class TestTransform(BaseTransform):
            name = "test"
            OUTPUT_SCHEMA: list[FieldDefinition] = []
            depends_on: list[str] = []  # Explicit declaration required

            def transform(self, dataset: VulnDataset) -> None:
                pass

        transform = TestTransform()
        assert transform.depends_on == []

    def test_base_transform_allows_depends_on_override(self) -> None:
        """Subclass can override depends_on."""

        class TestTransform(BaseTransform):
            name = "test"
            OUTPUT_SCHEMA: list[FieldDefinition] = []
            depends_on = ["kev"]

            def transform(self, dataset: VulnDataset) -> None:
                pass

        transform = TestTransform()
        assert transform.depends_on == ["kev"]

    def test_transform_missing_depends_on_raises(self) -> None:
        """Transform without explicit depends_on should fail."""

        with pytest.raises(
            AttributeError,
            match="must explicitly declare 'depends_on' class attribute",
        ):

            class BadTransform(BaseTransform):
                name: str = "BadTransform"
                OUTPUT_SCHEMA: list[FieldDefinition] = []
                # Missing: depends_on = []

                def transform(self, dataset: VulnDataset) -> None:
                    pass

            BadTransform()  # Should raise AttributeError

    def test_base_transform_has_default_schema_version(self) -> None:
        """BaseTransform provides default SCHEMA_VERSION."""

        class TestTransform(BaseTransform):
            name = "test"
            OUTPUT_SCHEMA: list[FieldDefinition] = []
            depends_on: list[str] = []

            def transform(self, dataset: VulnDataset) -> None:
                pass

        transform = TestTransform()
        assert transform.SCHEMA_VERSION == "1.0.0"

    def test_base_transform_allows_schema_version_override(self) -> None:
        """Subclass can override SCHEMA_VERSION."""

        class TestTransform(BaseTransform):
            name = "test"
            OUTPUT_SCHEMA: list[FieldDefinition] = []
            depends_on: list[str] = []
            SCHEMA_VERSION = "2.0.0"

            def transform(self, dataset: VulnDataset) -> None:
                pass

        transform = TestTransform()
        assert transform.SCHEMA_VERSION == "2.0.0"

    def test_base_transform_registers_schema_version(self) -> None:
        """BaseTransform registers schema with version in Registry."""
        Registry.clear()

        class TestTransform(BaseTransform):
            name = "VersionedTransform"
            OUTPUT_SCHEMA = [
                FieldDefinition("field", "str", "Test field", "VersionedTransform")
            ]
            depends_on: list[str] = []
            SCHEMA_VERSION = "1.2.3"

            def transform(self, dataset: VulnDataset) -> None:
                pass

        TestTransform()

        version = Registry.get_component_schema_version("VersionedTransform")
        assert version == "1.2.3"


class TestBaseEnricher:
    """Tests for BaseEnricher (async transforms)."""

    def test_base_enricher_is_abstract(self) -> None:
        """BaseEnricher cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract"):
            BaseEnricher()  # type: ignore[abstract]

    def test_base_enricher_inherits_from_base_transform(self) -> None:
        """BaseEnricher inherits from BaseTransform."""
        assert issubclass(BaseEnricher, BaseTransform)

    def test_base_enricher_requires_async_transform(self) -> None:
        """BaseEnricher requires async transform method."""

        class TestEnricher(BaseEnricher):
            name = "test"
            OUTPUT_SCHEMA: list[FieldDefinition] = []
            # Missing async def transform

        with pytest.raises(TypeError):
            TestEnricher()

    def test_base_enricher_accepts_async_transform(self) -> None:
        """BaseEnricher accepts async transform method."""

        class TestEnricher(BaseEnricher):
            name = "test"
            OUTPUT_SCHEMA: list[FieldDefinition] = []
            depends_on: list[str] = []

            async def transform(self, dataset: VulnDataset) -> None:
                pass

        enricher = TestEnricher()
        assert enricher is not None


class TestBaseInference:
    """Tests for BaseInference (sync with confidence)."""

    def test_base_inference_is_abstract(self) -> None:
        """BaseInference cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract"):
            BaseInference()  # type: ignore[abstract]

    def test_base_inference_inherits_from_base_transform(self) -> None:
        """BaseInference inherits from BaseTransform."""
        assert issubclass(BaseInference, BaseTransform)

    def test_base_inference_has_default_confidence_threshold(self) -> None:
        """BaseInference provides default confidence_threshold."""

        class TestInference(BaseInference):
            name = "test"
            OUTPUT_SCHEMA: list[FieldDefinition] = []
            depends_on: list[str] = []

            def transform(self, dataset: VulnDataset) -> None:
                pass

        inference = TestInference()
        assert inference.confidence_threshold == 0.7

    def test_base_inference_allows_confidence_override(self) -> None:
        """Subclass can override confidence_threshold."""

        class TestInference(BaseInference):
            name = "test"
            OUTPUT_SCHEMA: list[FieldDefinition] = []
            depends_on: list[str] = []
            confidence_threshold = 0.9

            def transform(self, dataset: VulnDataset) -> None:
                pass

        inference = TestInference()
        assert inference.confidence_threshold == 0.9


class TestBaseComputed:
    """Tests for BaseComputed (sync pure computation)."""

    def test_base_computed_is_abstract(self) -> None:
        """BaseComputed cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract"):
            BaseComputed()  # type: ignore[abstract]

    def test_base_computed_inherits_from_base_transform(self) -> None:
        """BaseComputed inherits from BaseTransform."""
        assert issubclass(BaseComputed, BaseTransform)

    def test_base_computed_requires_sync_transform(self) -> None:
        """BaseComputed requires sync transform method."""

        class TestComputed(BaseComputed):
            name = "test"
            OUTPUT_SCHEMA: list[FieldDefinition] = []
            depends_on: list[str] = []

            def transform(self, dataset: VulnDataset) -> None:
                pass

        computed = TestComputed()
        assert computed is not None
