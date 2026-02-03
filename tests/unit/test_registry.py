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
