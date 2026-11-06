"""
Registry - Plugin registration and lookup
Tests: @register_transform, get_transforms()

Layer: Core plugin system
Dependencies: None
"""

from __future__ import annotations

from vulnflow.registry import clear_registry, get_transforms, register_transform


def test_register_transform_decorator() -> None:
    """@register_transform registers transform class."""
    clear_registry()

    @register_transform
    class TestTransform:
        name = "test"
        depends_on: list[str] = []

        def transform(self, dataset):  # type: ignore[no-untyped-def]
            pass

    transforms = get_transforms()
    assert len(transforms) == 1
    assert transforms[0].__name__ == "TestTransform"


def test_get_transforms_returns_list() -> None:
    """get_transforms() returns list of registered transforms."""
    clear_registry()

    @register_transform
    class Transform1:
        name = "t1"
        depends_on: list[str] = []

    @register_transform
    class Transform2:
        name = "t2"
        depends_on: list[str] = []

    transforms = get_transforms()
    assert len(transforms) == 2


def test_clear_registry() -> None:
    """clear_registry() removes all registrations."""
    clear_registry()

    @register_transform
    class TestTransform:
        name = "test"
        depends_on: list[str] = []

    assert len(get_transforms()) == 1

    clear_registry()
    assert len(get_transforms()) == 0


def test_registry_register_extractor() -> None:
    """Registry.register_extractor() registers extractor class."""
    from vulnflow.registry import Registry

    clear_registry()

    @Registry.register_extractor("test_extractor")
    class TestExtractor:
        name = "test_extractor"
        format = "test"

        def extract(self):  # type: ignore[no-untyped-def]
            pass

    extractor_cls = Registry.get_extractor("test_extractor")
    assert extractor_cls is not None
    assert extractor_cls.__name__ == "TestExtractor"


def test_registry_get_extractor_not_found() -> None:
    """Registry.get_extractor() raises KeyError for unknown extractor."""
    import pytest

    from vulnflow.registry import Registry

    clear_registry()

    with pytest.raises(KeyError, match="unknown"):
        Registry.get_extractor("unknown")


def test_registry_list_extractors() -> None:
    """Registry.list_extractors() returns all registered extractor names."""
    from vulnflow.registry import Registry

    clear_registry()

    @Registry.register_extractor("extractor1")
    class Extractor1:
        name = "extractor1"
        format = "fmt1"

    @Registry.register_extractor("extractor2")
    class Extractor2:
        name = "extractor2"
        format = "fmt2"

    extractors = Registry.list_extractors()
    assert len(extractors) == 2
    assert "extractor1" in extractors
    assert "extractor2" in extractors


def test_registry_clear_extractors() -> None:
    """Registry.clear() clears extractor registrations."""
    from vulnflow.registry import Registry

    clear_registry()

    @Registry.register_extractor("test")
    class TestExtractor:
        name = "test"
        format = "test"

    assert len(Registry.list_extractors()) == 1

    clear_registry()
    assert len(Registry.list_extractors()) == 0


def test_registry_register_loader() -> None:
    """Registry.register_loader() registers loader class."""
    from vulnflow.registry import Registry

    clear_registry()

    @Registry.register_loader("test_loader")
    class TestLoader:
        name = "test_loader"
        format = "test"

        def load(self, dataset):  # type: ignore[no-untyped-def]
            pass

    loader_cls = Registry.get_loader("test_loader")
    assert loader_cls is not None
    assert loader_cls.__name__ == "TestLoader"


def test_registry_get_loader_not_found() -> None:
    """Registry.get_loader() raises KeyError for unknown loader."""
    import pytest

    from vulnflow.registry import Registry

    clear_registry()

    with pytest.raises(KeyError, match="unknown"):
        Registry.get_loader("unknown")


def test_registry_list_loaders() -> None:
    """Registry.list_loaders() returns all registered loader names."""
    from vulnflow.registry import Registry

    clear_registry()

    @Registry.register_loader("loader1")
    class Loader1:
        name = "loader1"
        format = "fmt1"

    @Registry.register_loader("loader2")
    class Loader2:
        name = "loader2"
        format = "fmt2"

    loaders = Registry.list_loaders()
    assert len(loaders) == 2
    assert "loader1" in loaders
    assert "loader2" in loaders


def test_registry_clear_loaders() -> None:
    """Registry.clear() clears loader registrations."""
    from vulnflow.registry import Registry

    clear_registry()

    @Registry.register_loader("test")
    class TestLoader:
        name = "test"
        format = "test"

    assert len(Registry.list_loaders()) == 1

    clear_registry()
    assert len(Registry.list_loaders()) == 0


def test_builtin_extractors_registered() -> None:
    """Built-in extractors are registered on module import."""
    # Import modules to trigger registration
    import vulnflow.extractors.files  # noqa: F401
    import vulnflow.extractors.nvd  # noqa: F401
    from vulnflow.registry import Registry

    extractors = Registry.list_extractors()
    assert "nvd_json" in extractors
    assert "parquet" in extractors

    # Verify classes can be retrieved
    nvd_cls = Registry.get_extractor("nvd_json")
    assert nvd_cls.__name__ == "NVDExtractor"

    parquet_cls = Registry.get_extractor("parquet")
    assert parquet_cls.__name__ == "ParquetExtractor"


def test_builtin_loaders_registered() -> None:
    """Built-in loaders are registered on module import."""
    # Import modules to trigger registration
    import vulnflow.loaders.dataframe  # noqa: F401
    import vulnflow.loaders.files  # noqa: F401
    from vulnflow.registry import Registry

    loaders = Registry.list_loaders()
    assert "parquet" in loaders
    assert "json" in loaders
    assert "pandas" in loaders
    assert "polars" in loaders

    # Verify classes can be retrieved
    parquet_cls = Registry.get_loader("parquet")
    assert parquet_cls.__name__ == "ParquetLoader"

    json_cls = Registry.get_loader("json")
    assert json_cls.__name__ == "JSONLoader"

    pandas_cls = Registry.get_loader("pandas")
    assert pandas_cls.__name__ == "PandasLoader"

    polars_cls = Registry.get_loader("polars")
    assert polars_cls.__name__ == "PolarsLoader"


# ============================================================================
# Unified registry tests
# ============================================================================


def test_registry_list_transforms() -> None:
    """Registry.list_transforms() returns all registered transform names."""
    from vulnflow.registry import Registry

    clear_registry()

    @Registry.register_transform("transform1")
    class Transform1:
        name = "transform1"

    @Registry.register_transform("transform2")
    class Transform2:
        name = "transform2"

    transforms = Registry.list_transforms()
    assert len(transforms) == 2
    assert "transform1" in transforms
    assert "transform2" in transforms


def test_registry_get_default_extractors() -> None:
    """Registry.get_default_extractors() returns empty list (no defaults)."""
    from vulnflow.registry import Registry

    clear_registry()

    @Registry.register_extractor("test")
    class TestExtractor:
        name = "test"
        format = "test"

    # No default extractors - returns empty list
    defaults = Registry.get_default_extractors()
    assert defaults == []


def test_registry_get_default_loaders() -> None:
    """Registry.get_default_loaders() returns empty list (no defaults)."""
    from vulnflow.registry import Registry

    clear_registry()

    @Registry.register_loader("test")
    class TestLoader:
        name = "test"
        format = "test"

    # No default loaders - returns empty list
    defaults = Registry.get_default_loaders()
    assert defaults == []


def test_registry_schema_versioning() -> None:
    """Registry.register_schema() stores version with fields."""
    from vulnflow.registry import FieldDefinition, Registry

    clear_registry()

    fields = [FieldDefinition("test_field", "str", "Test field", "TestComponent")]
    Registry.register_schema("TestComponent", fields, version="1.2.3")

    # Verify version is stored
    version = Registry.get_component_schema_version("TestComponent")
    assert version == "1.2.3"

    # Verify fields are still retrievable
    retrieved_fields = Registry.get_component_columns("TestComponent")
    assert len(retrieved_fields) == 1
    assert retrieved_fields[0].name == "test_field"


def test_registry_schema_default_version() -> None:
    """Registry.register_schema() defaults to version 1.0.0."""
    from vulnflow.registry import FieldDefinition, Registry

    clear_registry()

    fields = [FieldDefinition("test_field", "str", "Test field", "TestComponent")]
    Registry.register_schema("TestComponent", fields)

    version = Registry.get_component_schema_version("TestComponent")
    assert version == "1.0.0"


def test_registry_get_component_schema_version_not_found() -> None:
    """Registry.get_component_schema_version() raises KeyError for unknown component."""
    import pytest

    from vulnflow.registry import Registry

    clear_registry()

    with pytest.raises(KeyError, match="unknown"):
        Registry.get_component_schema_version("unknown")


def test_registry_get_all_columns_with_versioning() -> None:
    """Registry.get_all_columns() works with versioned schema storage."""
    from vulnflow.registry import FieldDefinition, Registry

    clear_registry()

    fields1 = [FieldDefinition("field1", "str", "Field 1", "Component1")]
    fields2 = [FieldDefinition("field2", "int", "Field 2", "Component2")]

    Registry.register_schema("Component1", fields1, version="1.0.0")
    Registry.register_schema("Component2", fields2, version="2.0.0")

    all_fields = Registry.get_all_columns()
    assert len(all_fields) == 2
    field_names = [f.name for f in all_fields]
    assert "field1" in field_names
    assert "field2" in field_names


def test_registry_get_transform_raises_keyerror() -> None:
    """Registry.get_transform() raises KeyError for unknown transform."""
    import pytest

    from vulnflow.registry import Registry

    clear_registry()

    with pytest.raises(KeyError, match="unknown"):
        Registry.get_transform("unknown")


# =============================================================================
# FieldDefinition and Schema Registration Tests
# =============================================================================


def test_field_definition_creation() -> None:
    """FieldDefinition dataclass can be created."""
    from vulnflow.registry import FieldDefinition

    field = FieldDefinition(
        name="kev_date_added",
        type="datetime",
        description="Date added to KEV catalog",
        source_component="KEVEnricher",
    )

    assert field.name == "kev_date_added"
    assert field.type == "datetime"
    assert field.description == "Date added to KEV catalog"
    assert field.source_component == "KEVEnricher"


def test_register_schema() -> None:
    """Registry.register_schema() stores schema."""
    from vulnflow.registry import FieldDefinition, Registry

    Registry.clear()

    fields = [
        FieldDefinition("kev_date_added", "datetime", "KEV date added", "KEVEnricher")
    ]

    Registry.register_schema("KEVEnricher", fields)

    # Should be retrievable
    result = Registry.get_component_columns("KEVEnricher")
    assert len(result) == 1
    assert result[0].name == "kev_date_added"


def test_get_all_columns() -> None:
    """Registry.get_all_columns() returns all registered fields."""
    from vulnflow.registry import FieldDefinition, Registry

    Registry.clear()

    kev_fields = [
        FieldDefinition("kev_date_added", "datetime", "KEV date", "KEVEnricher")
    ]
    epss_fields = [
        FieldDefinition("epss_probability", "float", "EPSS probability", "EPSSEnricher")
    ]

    Registry.register_schema("KEVEnricher", kev_fields)
    Registry.register_schema("EPSSEnricher", epss_fields)

    all_fields = Registry.get_all_columns()

    assert len(all_fields) == 2
    field_names = {f.name for f in all_fields}
    assert "kev_date_added" in field_names
    assert "epss_probability" in field_names


def test_get_component_columns_nonexistent() -> None:
    """Registry.get_component_columns() returns empty list for unknown component."""
    from vulnflow.registry import Registry

    Registry.clear()

    result = Registry.get_component_columns("NonExistent")

    assert result == []


# =============================================================================
# Transform Schema Registration Tests
# =============================================================================


def test_kev_enricher_has_output_schema() -> None:
    """KEVEnricher defines OUTPUT_SCHEMA with FieldDefinitions."""
    from vulnflow.registry import FieldDefinition
    from vulnflow.transforms.enrichers.kev import KEVEnricher

    assert hasattr(KEVEnricher, "OUTPUT_SCHEMA")
    assert isinstance(KEVEnricher.OUTPUT_SCHEMA, list)
    assert len(KEVEnricher.OUTPUT_SCHEMA) > 0

    # Verify schema structure (new FieldDefinition format)
    for field in KEVEnricher.OUTPUT_SCHEMA:
        assert isinstance(field, FieldDefinition)
        assert field.name
        assert field.type


def test_epss_enricher_has_output_schema() -> None:
    """EPSSEnricher defines OUTPUT_SCHEMA with FieldDefinitions."""
    from vulnflow.registry import FieldDefinition
    from vulnflow.transforms.enrichers.epss import EPSSEnricher

    assert hasattr(EPSSEnricher, "OUTPUT_SCHEMA")
    assert isinstance(EPSSEnricher.OUTPUT_SCHEMA, list)
    assert len(EPSSEnricher.OUTPUT_SCHEMA) == 1  # epss_scores JSON column

    # Verify schema structure (new FieldDefinition format)
    for field in EPSSEnricher.OUTPUT_SCHEMA:
        assert isinstance(field, FieldDefinition)
        assert field.name
        assert field.type

    # Verify the single field is epss_scores
    assert EPSSEnricher.OUTPUT_SCHEMA[0].name == "epss_scores"
    assert EPSSEnricher.OUTPUT_SCHEMA[0].type == "str"


def test_event_inference_has_schema() -> None:
    """EventInference defines SCHEMA_VERSION and OUTPUT_SCHEMA."""
    from vulnflow.transforms.inferences.events import EventInference

    assert hasattr(EventInference, "SCHEMA_VERSION")
    assert hasattr(EventInference, "OUTPUT_SCHEMA")
    assert len(EventInference.OUTPUT_SCHEMA) >= 6  # 6 CVD timestamp columns


def test_tag_inference_has_schema() -> None:
    """TagInference defines SCHEMA_VERSION and OUTPUT_SCHEMA."""
    from vulnflow.transforms.inferences.tags import TagInference

    assert hasattr(TagInference, "SCHEMA_VERSION")
    assert hasattr(TagInference, "OUTPUT_SCHEMA")


def test_cvd_state_transform_has_schema() -> None:
    """CVDStateTransform defines SCHEMA_VERSION and OUTPUT_SCHEMA with all 19 fields."""
    from vulnflow.transforms.computed.cvd_state import CVDStateTransform

    assert hasattr(CVDStateTransform, "SCHEMA_VERSION")
    assert hasattr(CVDStateTransform, "OUTPUT_SCHEMA")
    # CVDStateTransform now computes all 19 CVD fields (including desiderata)
    assert len(CVDStateTransform.OUTPUT_SCHEMA) == 19


def test_cvd_state_transform_includes_desiderata_fields() -> None:
    """CVDStateTransform OUTPUT_SCHEMA includes desiderata_mask and desiderata_count."""
    from vulnflow.transforms.computed.cvd_state import CVDStateTransform

    field_names = [f.name for f in CVDStateTransform.OUTPUT_SCHEMA]

    # Core fields
    assert "cvd_state" in field_names
    assert "state_bits" in field_names

    # Desiderata fields (consolidated from old CVDDesiderataTransform)
    assert "desiderata_mask" in field_names
    assert "desiderata_count" in field_names

    # Zero-day flags
    assert "is_zero_day_exploit" in field_names
    assert "is_zero_day_attack" in field_names


# =============================================================================
# Schema Introspection Utilities Tests
# =============================================================================


def test_get_record_columns_returns_list() -> None:
    """get_record_columns() returns list of field names."""
    from vulnflow.utils.schema import get_record_columns

    columns = get_record_columns()

    assert isinstance(columns, list)
    assert len(columns) > 0


def test_get_record_columns_contains_id() -> None:
    """get_record_columns() includes VulnRecord fields."""
    from vulnflow.utils.schema import get_record_columns

    columns = get_record_columns()

    assert "id" in columns
    assert "cve_id" in columns
    assert "description" in columns
    assert "cvss_scores" in columns


def test_get_flattened_columns_returns_list() -> None:
    """get_flattened_columns() returns list of flattened column names."""
    from vulnflow.utils.schema import get_flattened_columns

    columns = get_flattened_columns()

    assert isinstance(columns, list)
    assert len(columns) > 0


def test_get_flattened_columns_contains_cvss() -> None:
    """get_flattened_columns() includes CVSS flattened columns."""
    from vulnflow.utils.schema import get_flattened_columns

    columns = get_flattened_columns()

    assert "cvss_base_score" in columns
    assert "cvss_base_severity" in columns
    assert "cvss_version" in columns
    assert "cvss_vector_string" in columns
    assert "cvss_source" in columns


def test_get_flattened_columns_contains_cwe() -> None:
    """get_flattened_columns() includes CWE flattened columns."""
    from vulnflow.utils.schema import get_flattened_columns

    columns = get_flattened_columns()

    assert "primary_cwe_id" in columns
    assert "primary_cwe_name" in columns


def test_get_flattened_columns_contains_cpe() -> None:
    """get_flattened_columns() includes CPE flattened columns."""
    from vulnflow.utils.schema import get_flattened_columns

    columns = get_flattened_columns()

    assert "primary_vendor" in columns
    assert "primary_product" in columns
    assert "primary_version" in columns


# =============================================================================
# FieldType Enum Tests
# =============================================================================


def test_field_type_is_strenum() -> None:
    """FieldType members are usable as plain strings."""
    from vulnflow.registry import FieldType

    assert FieldType.STR == "str"
    assert FieldType.DATETIME == "datetime"
    assert FieldType.UUID == "uuid"
    assert FieldType.LIST_STR == "list[str]"


def test_field_type_in_field_definition() -> None:
    """FieldDefinition.type accepts FieldType members."""
    from vulnflow.registry import FieldDefinition, FieldType

    fd = FieldDefinition(
        name="test",
        type=FieldType.DATETIME,
        description="Test field",
    )
    assert fd.type == "datetime"
    assert fd.type == FieldType.DATETIME


def test_field_type_backward_compatible() -> None:
    """FieldDefinition.type still accepts raw strings."""
    from vulnflow.registry import FieldDefinition, FieldType

    fd = FieldDefinition(name="test", type="str", description="Test field")
    assert fd.type == FieldType.STR


def test_field_type_all_members() -> None:
    """FieldType has all expected members."""
    from vulnflow.registry import FieldType

    expected = {
        "str",
        "bool",
        "int",
        "float",
        "float32",
        "uint8",
        "uint16",
        "datetime",
        "uuid",
        "list[str]",
    }
    actual = {member.value for member in FieldType}
    assert actual == expected


# =============================================================================
# Extended FieldDefinition Tests
# =============================================================================


def test_field_definition_nullable_default() -> None:
    """FieldDefinition.nullable defaults to True."""
    from vulnflow.registry import FieldDefinition

    fd = FieldDefinition(name="test", type="str", description="Test")
    assert fd.nullable is True


def test_field_definition_is_json_default() -> None:
    """FieldDefinition.is_json defaults to False."""
    from vulnflow.registry import FieldDefinition

    fd = FieldDefinition(name="test", type="str", description="Test")
    assert fd.is_json is False


def test_field_definition_with_new_fields() -> None:
    """FieldDefinition accepts nullable and is_json."""
    from vulnflow.registry import FieldDefinition

    fd = FieldDefinition(
        name="data",
        type="str",
        description="JSON column",
        nullable=False,
        is_json=True,
    )
    assert fd.nullable is False
    assert fd.is_json is True


def test_field_definition_backward_compat_no_new_args() -> None:
    """Existing FieldDefinition calls without new args still work."""
    from vulnflow.registry import FieldDefinition

    fd = FieldDefinition("kev_date", "datetime", "KEV date", "KEVEnricher")
    assert fd.name == "kev_date"
    assert fd.type == "datetime"
    assert fd.source_component == "KEVEnricher"
    assert fd.nullable is True
    assert fd.is_json is False


# =============================================================================
# get_full_schema() Tests
# =============================================================================


def test_get_full_schema_includes_base_columns() -> None:
    """get_full_schema() includes VulnRecord base columns."""
    from vulnflow.registry import Registry

    clear_registry()
    schema = Registry.get_full_schema()
    names = {fd.name for fd in schema}
    assert "id" in names
    assert "cve_id" in names
    assert "cvss_scores" in names


def test_get_full_schema_includes_transform_columns() -> None:
    """get_full_schema() includes registered transform columns."""
    from vulnflow.registry import FieldDefinition, Registry

    clear_registry()
    Registry.register_schema(
        "TestTransform",
        [FieldDefinition("custom_field", "float", "Custom", "TestTransform")],
    )
    schema = Registry.get_full_schema()
    names = {fd.name for fd in schema}
    assert "custom_field" in names
    assert "id" in names  # base still present


def test_get_full_schema_deduplicates_by_name() -> None:
    """get_full_schema() deduplicates — transform overrides base."""
    from vulnflow.registry import FieldDefinition, Registry

    clear_registry()
    # Register a transform that redefines "cve_id" with different type
    Registry.register_schema(
        "Override",
        [FieldDefinition("cve_id", "int", "Overridden", "Override", nullable=False)],
    )
    schema = Registry.get_full_schema()
    cve_fields = [fd for fd in schema if fd.name == "cve_id"]
    assert len(cve_fields) == 1
    assert cve_fields[0].source_component == "Override"
    assert cve_fields[0].type == "int"


def test_get_full_schema_base_first_order() -> None:
    """get_full_schema() returns base columns first, then transforms."""
    from vulnflow.registry import FieldDefinition, Registry

    clear_registry()
    Registry.register_schema(
        "TestTransform",
        [FieldDefinition("zzz_transform_col", "str", "Last", "TestTransform")],
    )
    schema = Registry.get_full_schema()
    names = [fd.name for fd in schema]
    # id should come before zzz_transform_col
    assert names.index("id") < names.index("zzz_transform_col")


def test_get_full_schema_lazy_base_registration() -> None:
    """get_full_schema() lazily registers _base schema on first call."""
    from vulnflow.registry import Registry

    clear_registry()
    assert "_base" not in Registry._schemas
    Registry.get_full_schema()
    assert "_base" in Registry._schemas


# =============================================================================
# get_schema_for_columns() Tests
# =============================================================================


def test_get_schema_for_columns_known() -> None:
    """get_schema_for_columns() returns FieldDefinitions for known columns."""
    from vulnflow.registry import Registry

    clear_registry()
    result = Registry.get_schema_for_columns(["id", "cve_id"])
    assert len(result) == 2
    assert result[0].name == "id"
    assert result[1].name == "cve_id"


def test_get_schema_for_columns_preserves_order() -> None:
    """get_schema_for_columns() returns results in input order."""
    from vulnflow.registry import Registry

    clear_registry()
    result = Registry.get_schema_for_columns(["cve_id", "id", "description"])
    assert [fd.name for fd in result] == ["cve_id", "id", "description"]


def test_get_schema_for_columns_unknown_inferred() -> None:
    """get_schema_for_columns() infers unregistered columns as str/nullable."""
    from vulnflow.registry import Registry

    clear_registry()
    result = Registry.get_schema_for_columns(["totally_unknown_col"])
    assert len(result) == 1
    assert result[0].name == "totally_unknown_col"
    assert result[0].type == "str"
    assert result[0].nullable is True


def test_get_schema_for_columns_empty_input() -> None:
    """get_schema_for_columns() returns empty list for empty input."""
    from vulnflow.registry import Registry

    clear_registry()
    result = Registry.get_schema_for_columns([])
    assert result == []


# =============================================================================
# FieldType Export Tests
# =============================================================================


def test_field_type_exported_from_vulnflow() -> None:
    """FieldType is importable from vulnflow top-level."""
    from vulnflow import FieldType

    assert FieldType.STR == "str"
