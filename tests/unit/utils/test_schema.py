"""Tests for schema introspection and type conversion utilities."""

from datetime import datetime
from uuid import UUID

import pytest

from vulnflow.utils.schema import (
    build_type_map,
    deserialize_value,
    get_flattened_columns,
    get_record_columns,
    infer_type,
    serialize_value,
)


# === Introspection Tests (existing functionality) ===


def test_get_record_columns_returns_list():
    """get_record_columns returns a list."""
    cols = get_record_columns()
    assert isinstance(cols, list)
    assert len(cols) > 0


def test_get_record_columns_contains_core_fields():
    """get_record_columns includes core VulnRecord fields."""
    cols = get_record_columns()
    assert "cve_id" in cols
    assert "description" in cols
    assert "cvss_scores" in cols
    assert "cwes" in cols


def test_get_flattened_columns_returns_expected_count():
    """get_flattened_columns returns 10 columns."""
    cols = get_flattened_columns()
    assert len(cols) == 10


def test_get_flattened_columns_contains_cvss_fields():
    """get_flattened_columns includes CVSS flattened fields."""
    cols = get_flattened_columns()
    assert "cvss_base_score" in cols
    assert "cvss_base_severity" in cols
    assert "cvss_version" in cols


def test_get_flattened_columns_contains_cwe_fields():
    """get_flattened_columns includes CWE flattened fields."""
    cols = get_flattened_columns()
    assert "primary_cwe_id" in cols
    assert "primary_cwe_name" in cols


def test_get_flattened_columns_contains_cpe_fields():
    """get_flattened_columns includes CPE flattened fields."""
    cols = get_flattened_columns()
    assert "primary_vendor" in cols
    assert "primary_product" in cols
    assert "primary_version" in cols


# === Type Conversion Tests (new functionality) ===


def test_build_type_map_returns_dict():
    """build_type_map returns a dictionary."""
    type_map = build_type_map()
    assert isinstance(type_map, dict)


def test_build_type_map_contains_core_types():
    """build_type_map includes types for core fields."""
    from vulnflow.registry import FieldDefinition, Registry

    # Register test fields
    Registry.register_schema(
        "TestSchema",
        [
            FieldDefinition("id", "uuid", "Test UUID field"),
            FieldDefinition("cve_id", "str", "Test string field"),
        ],
    )

    type_map = build_type_map()
    assert "id" in type_map
    assert type_map["id"] == "uuid"
    assert "cve_id" in type_map
    assert type_map["cve_id"] == "str"


def test_deserialize_value_none():
    """deserialize_value handles None and 'None' string."""
    type_map = build_type_map()

    assert deserialize_value(None, "cve_id", type_map) is None
    assert deserialize_value("None", "cve_id", type_map) is None


def test_deserialize_value_datetime():
    """deserialize_value converts ISO datetime string to datetime."""
    from vulnflow.registry import FieldDefinition, Registry

    # Register datetime field
    Registry.register_schema(
        "TestDatetime",
        [FieldDefinition("nvd_published_at", "datetime", "Test datetime field")],
    )

    type_map = build_type_map()
    result = deserialize_value("2024-01-15T10:30:00", "nvd_published_at", type_map)

    assert isinstance(result, datetime)
    assert result.year == 2024
    assert result.month == 1
    assert result.day == 15
    assert result.hour == 10
    assert result.minute == 30


def test_deserialize_value_uuid():
    """deserialize_value converts UUID string to UUID object."""
    from vulnflow.registry import FieldDefinition, Registry

    # Register UUID field
    Registry.register_schema(
        "TestUUID",
        [FieldDefinition("id", "uuid", "Test UUID field")],
    )

    type_map = build_type_map()
    uuid_str = "550e8400-e29b-41d4-a716-446655440000"

    result = deserialize_value(uuid_str, "id", type_map)

    assert isinstance(result, UUID)
    assert str(result) == uuid_str


def test_deserialize_value_int():
    """deserialize_value converts string to int."""
    type_map = {"count": "int"}

    result = deserialize_value("42", "count", type_map)

    assert isinstance(result, int)
    assert result == 42


def test_deserialize_value_float():
    """deserialize_value converts string to float."""
    type_map = {"score": "float"}

    result = deserialize_value("3.14", "score", type_map)

    assert isinstance(result, float)
    assert result == 3.14


def test_deserialize_value_bool_true():
    """deserialize_value converts 'True' string to bool."""
    type_map = {"flag": "bool"}

    result = deserialize_value("True", "flag", type_map)

    assert isinstance(result, bool)
    assert result is True


def test_deserialize_value_bool_false():
    """deserialize_value converts 'False' string to bool."""
    type_map = {"flag": "bool"}

    result = deserialize_value("False", "flag", type_map)

    assert isinstance(result, bool)
    assert result is False


def test_deserialize_value_str():
    """deserialize_value keeps string as-is."""
    type_map = {"text": "str"}

    result = deserialize_value("hello", "text", type_map)

    assert isinstance(result, str)
    assert result == "hello"


def test_deserialize_value_unknown_type_infers():
    """deserialize_value falls back to infer_type for unknown columns."""
    type_map = {}

    # Should infer int
    result = deserialize_value("123", "unknown_col", type_map)
    assert result == 123
    assert isinstance(result, int)


def test_serialize_value_none():
    """serialize_value keeps None as None."""
    result = serialize_value(None)
    assert result is None


def test_serialize_value_str():
    """serialize_value keeps string as-is."""
    result = serialize_value("hello")
    assert result == "hello"


def test_serialize_value_int():
    """serialize_value keeps int as-is."""
    result = serialize_value(42)
    assert result == 42


def test_serialize_value_float():
    """serialize_value keeps float as-is."""
    result = serialize_value(3.14)
    assert result == 3.14


def test_serialize_value_bool():
    """serialize_value keeps bool as-is."""
    result = serialize_value(True)
    assert result is True


def test_serialize_value_datetime():
    """serialize_value converts datetime to ISO format string."""
    dt = datetime(2024, 1, 15, 10, 30, 0)

    result = serialize_value(dt)

    assert isinstance(result, str)
    assert result == "2024-01-15T10:30:00"


def test_serialize_value_uuid():
    """serialize_value converts UUID to string."""
    uuid_obj = UUID("550e8400-e29b-41d4-a716-446655440000")

    result = serialize_value(uuid_obj)

    assert isinstance(result, str)
    assert result == "550e8400-e29b-41d4-a716-446655440000"


def test_serialize_value_unknown_object():
    """serialize_value converts unknown objects to string."""
    class CustomObj:
        def __str__(self):
            return "custom"

    result = serialize_value(CustomObj())

    assert isinstance(result, str)
    assert result == "custom"


def test_infer_type_positive_int():
    """infer_type converts positive int string to int."""
    result = infer_type("123")
    assert result == 123
    assert isinstance(result, int)


def test_infer_type_negative_int():
    """infer_type converts negative int string to int."""
    result = infer_type("-456")
    assert result == -456
    assert isinstance(result, int)


def test_infer_type_float():
    """infer_type converts float string to float."""
    result = infer_type("3.14")
    assert result == 3.14
    assert isinstance(result, float)


def test_infer_type_bool_true():
    """infer_type converts 'True' to bool."""
    result = infer_type("True")
    assert result is True
    assert isinstance(result, bool)


def test_infer_type_bool_false():
    """infer_type converts 'False' to bool."""
    result = infer_type("False")
    assert result is False
    assert isinstance(result, bool)


def test_infer_type_keeps_string():
    """infer_type keeps non-convertible strings as-is."""
    result = infer_type("hello world")
    assert result == "hello world"
    assert isinstance(result, str)


def test_infer_type_non_string():
    """infer_type keeps non-string values as-is."""
    result = infer_type(42)
    assert result == 42


def test_roundtrip_datetime():
    """Datetime survives serialize → deserialize roundtrip."""
    from vulnflow.registry import FieldDefinition, Registry

    # Register datetime field
    Registry.register_schema(
        "TestRoundtripDatetime",
        [FieldDefinition("nvd_published_at", "datetime", "Test datetime field")],
    )

    type_map = build_type_map()
    original = datetime(2024, 1, 15, 10, 30, 0)

    serialized = serialize_value(original)
    deserialized = deserialize_value(serialized, "nvd_published_at", type_map)

    assert deserialized == original


def test_roundtrip_uuid():
    """UUID survives serialize → deserialize roundtrip."""
    from vulnflow.registry import FieldDefinition, Registry

    # Register UUID field
    Registry.register_schema(
        "TestRoundtripUUID",
        [FieldDefinition("id", "uuid", "Test UUID field")],
    )

    type_map = build_type_map()
    original = UUID("550e8400-e29b-41d4-a716-446655440000")

    serialized = serialize_value(original)
    deserialized = deserialize_value(serialized, "id", type_map)

    assert deserialized == original


def test_roundtrip_primitives():
    """Primitives survive serialize → deserialize roundtrip."""
    type_map = {"text": "str", "num": "int", "score": "float", "flag": "bool"}

    # String
    assert deserialize_value(serialize_value("hello"), "text", type_map) == "hello"

    # Int
    assert deserialize_value(serialize_value(42), "num", type_map) == 42

    # Float
    assert deserialize_value(serialize_value(3.14), "score", type_map) == 3.14

    # Bool
    assert deserialize_value(serialize_value(True), "flag", type_map) is True

    # None
    assert deserialize_value(serialize_value(None), "text", type_map) is None
