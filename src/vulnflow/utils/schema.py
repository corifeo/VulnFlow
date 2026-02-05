"""Schema introspection and type conversion utilities.

Provides:
- get_record_columns: Get VulnRecord field names
- get_flattened_columns: Get flattened column names for DataFrame export
- build_type_map: Build column name → type mapping from Registry
- deserialize_value: JSON → Python type using schema
- serialize_value: Python type → JSON-serializable
- infer_type: Fallback type inference for unregistered fields

Layer: Utility
Dependencies: schema, registry
Used by: extractors, loaders, transforms
"""

from __future__ import annotations

from dataclasses import fields
from datetime import datetime
from typing import Any
from uuid import UUID


def get_record_columns() -> list[str]:
    """Get column names from VulnRecord dataclass.

    Extracts field names using dataclasses.fields() introspection.
    Used by extractors and loaders to ensure schema consistency.

    Returns:
        List of field names from VulnRecord schema in declaration order

    Examples:
        >>> cols = get_record_columns()
        >>> "cve_id" in cols
        True
        >>> "cvss_scores" in cols
        True
    """
    from vulnflow.schema import VulnRecord

    return [f.name for f in fields(VulnRecord)]


def get_flattened_columns() -> list[str]:
    """Get computed/flattened column names for DataFrame export.

    These columns are derived from VulnRecord nested fields during extraction,
    following flattening rules from DATA-STRUCTURES.md:
    - cvss_scores → cvss_base_score, cvss_version, cvss_vector_string, etc.
    - cwes → primary_cwe_id, primary_cwe_name
    - cpes → primary_vendor, primary_product, primary_version

    Returns:
        List of flattened column names in canonical order

    Examples:
        >>> cols = get_flattened_columns()
        >>> "cvss_base_score" in cols
        True
        >>> "primary_cwe_id" in cols
        True
        >>> len(cols)
        10
    """
    return [
        # CVSS flattened (from cvss_scores)
        "cvss_base_score",
        "cvss_base_severity",
        "cvss_version",
        "cvss_vector_string",
        "cvss_source",
        # CWE flattened (from cwes)
        "primary_cwe_id",
        "primary_cwe_name",
        # CPE flattened (from cpes)
        "primary_vendor",
        "primary_product",
        "primary_version",
    ]


# === Type Conversion ===


def build_type_map() -> dict[str, str]:
    """Build column name → type mapping from Registry schema.

    Queries Registry.get_all_columns() to extract type information
    for all registered fields.

    Returns:
        Dictionary mapping column names to type strings
        Example: {"cve_id": "str", "nvd_published_at": "datetime", "id": "uuid"}

    Examples:
        >>> type_map = build_type_map()
        >>> type_map["nvd_published_at"]
        "datetime"
        >>> type_map["id"]
        "uuid"
    """
    from vulnflow.registry import Registry

    type_map = {}
    for field_def in Registry.get_all_columns():
        type_map[field_def.name] = field_def.type
    return type_map


def deserialize_value(value: Any, column: str, type_map: dict[str, str]) -> Any:
    """Convert JSON-serializable value → Python type using schema.

    Handles:
    - datetime: ISO format string → datetime object
    - uuid: String → UUID object
    - int/float/bool: String → proper type
    - None/"None": → None
    - Unknown types: Falls back to infer_type()

    Args:
        value: Value to convert (usually string from JSON)
        column: Column name for schema lookup
        type_map: Mapping from build_type_map()

    Returns:
        Value converted to appropriate Python type

    Examples:
        >>> type_map = build_type_map()
        >>> deserialize_value("2024-01-15T10:30:00", "nvd_published_at", type_map)
        datetime(2024, 1, 15, 10, 30, 0)
        >>> deserialize_value("550e8400-e29b-41d4-a716-446655440000", "id", type_map)
        UUID('550e8400-e29b-41d4-a716-446655440000')
        >>> deserialize_value("None", "description", type_map)
        None
    """
    if value is None or value == "None":
        return None

    # Get expected type from schema
    expected_type = type_map.get(column)

    if expected_type == "datetime":
        return datetime.fromisoformat(value)
    elif expected_type == "uuid":
        return UUID(value)
    elif expected_type == "int":
        return int(value)
    elif expected_type == "float":
        return float(value)
    elif expected_type == "bool":
        # Handle both bool (already deserialized) and string representation
        if isinstance(value, bool):
            return value
        return value == "True"
    elif expected_type == "str":
        return value
    else:
        # Fallback: basic inference for unregistered fields
        return infer_type(value)


def serialize_value(value: Any) -> Any:
    """Convert Python type → JSON-serializable format.

    Handles:
    - None: → None (kept as-is)
    - str/int/float/bool: → kept as-is (already JSON-serializable)
    - datetime: → ISO format string
    - UUID: → string
    - Other objects: → str(value)

    Args:
        value: Python value to serialize

    Returns:
        JSON-serializable value (str, int, float, bool, None)

    Examples:
        >>> serialize_value(datetime(2024, 1, 15, 10, 30))
        "2024-01-15T10:30:00"
        >>> serialize_value(UUID('550e8400-e29b-41d4-a716-446655440000'))
        "550e8400-e29b-41d4-a716-446655440000"
        >>> serialize_value(42)
        42
        >>> serialize_value(None)
        None
    """
    if value is None:
        return None

    # Already JSON-serializable
    if isinstance(value, (str, int, float, bool)):
        return value

    # Convert datetime to ISO format
    if isinstance(value, datetime):
        return value.isoformat()

    # Convert UUID to string
    if isinstance(value, UUID):
        return str(value)

    # Fallback: convert to string
    return str(value)


def infer_type(value: Any) -> Any:
    """Fallback type inference for unregistered fields.

    Used when column is not in Registry schema. Attempts basic
    type inference from string values.

    Tries (in order):
    1. int: "123" or "-456"
    2. float: "3.14"
    3. bool: "True" or "False"
    4. Keep as-is if none match

    Args:
        value: Value to infer type for

    Returns:
        Value with inferred type

    Examples:
        >>> infer_type("123")
        123
        >>> infer_type("3.14")
        3.14
        >>> infer_type("True")
        True
        >>> infer_type("unknown")
        "unknown"
    """
    if not isinstance(value, str):
        return value

    # Try int
    if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
        return int(value)

    # Try float
    try:
        return float(value)
    except ValueError:
        pass

    # Try bool
    if value in ("True", "False"):
        return value == "True"

    # Keep as-is
    return value
