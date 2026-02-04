"""
JSON Utilities - Tests for orjson wrapper functions
Tests: dumps, loads, dump, load

Layer: Utils/JSON
Dependencies: orjson
"""

from __future__ import annotations

from pathlib import Path

from vulnflow.utils import json


def test_dumps_returns_string():
    """dumps() returns a string, not bytes."""
    obj = {"key": "value", "number": 42}
    result = json.dumps(obj)

    assert isinstance(result, str)
    assert '"key":"value"' in result or '"key": "value"' in result


def test_loads_accepts_string():
    """loads() accepts a string."""
    s = '{"key": "value"}'
    result = json.loads(s)

    assert result == {"key": "value"}


def test_loads_accepts_bytes():
    """loads() accepts bytes."""
    b = b'{"key": "value"}'
    result = json.loads(b)

    assert result == {"key": "value"}


def test_dumps_with_indent():
    """dumps() with indent=True formats output."""
    obj = {"key": "value"}
    result = json.dumps(obj, indent=True)

    assert isinstance(result, str)
    # Should contain newlines when indented
    assert "\n" in result


def test_dumps_without_indent():
    """dumps() without indent produces compact output."""
    obj = {"key": "value"}
    result = json.dumps(obj)

    # Should not contain newlines
    assert "\n" not in result


def test_dump_to_file(tmp_path: Path):
    """dump() writes JSON to file."""
    obj = {"test": "data", "value": 123}
    file_path = tmp_path / "test.json"

    with open(file_path, "w") as f:
        json.dump(obj, f)

    # Read back and verify
    with open(file_path) as f:
        content = f.read()

    result = json.loads(content)
    assert result == obj


def test_dump_with_indent(tmp_path: Path):
    """dump() with indent=True writes formatted JSON."""
    obj = {"key": "value"}
    file_path = tmp_path / "test.json"

    with open(file_path, "w") as f:
        json.dump(obj, f, indent=True)

    with open(file_path) as f:
        content = f.read()

    # Indented output should have newlines
    assert "\n" in content


def test_load_from_file(tmp_path: Path):
    """load() reads JSON from file."""
    obj = {"test": "data", "value": 123}
    file_path = tmp_path / "test.json"

    # Write file
    with open(file_path, "w") as f:
        json.dump(obj, f)

    # Read back
    with open(file_path) as f:
        result = json.load(f)

    assert result == obj


def test_load_from_binary_file(tmp_path: Path):
    """load() works with binary file handles."""
    obj = {"test": "data"}
    file_path = tmp_path / "test.json"

    # Write JSON
    with open(file_path, "w") as f:
        json.dump(obj, f)

    # Read as binary
    with open(file_path, "rb") as f:
        result = json.load(f)

    assert result == obj


def test_dumps_handles_dataclass():
    """dumps() can serialize dataclasses."""
    from dataclasses import dataclass

    @dataclass
    class TestData:
        name: str
        value: int

    obj = TestData(name="test", value=42)
    result = json.dumps(obj)

    assert isinstance(result, str)
    # Should contain the field values
    assert "test" in result
    assert "42" in result


def test_roundtrip_preserves_data():
    """Roundtrip dumps/loads preserves data."""
    obj = {
        "string": "hello",
        "number": 123,
        "float": 3.14,
        "bool": True,
        "null": None,
        "list": [1, 2, 3],
        "nested": {"key": "value"},
    }

    serialized = json.dumps(obj)
    result = json.loads(serialized)

    assert result == obj
