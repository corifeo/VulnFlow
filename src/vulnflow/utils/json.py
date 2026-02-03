"""Fast JSON serialization using orjson.

This module provides a drop-in replacement for stdlib json with orjson backend.
All functions return/accept strings (not bytes) for API compatibility.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import IO, Any

import orjson

# Re-export JSONDecodeError for exception handling compatibility
JSONDecodeError = orjson.JSONDecodeError


def dumps(
    obj: Any,
    *,
    default: Callable[[Any], Any] | None = None,
    indent: bool = False,
    ensure_ascii: bool = True,  # ignored, orjson always outputs UTF-8
) -> str:
    """Serialize obj to a JSON string.

    Args:
        obj: Object to serialize.
        default: Function for objects that can't be serialized.
        indent: If True, format with 2-space indentation.
        ensure_ascii: Ignored (kept for stdlib json compatibility).

    Returns:
        JSON string.
    """
    opts = orjson.OPT_NON_STR_KEYS
    if indent:
        opts |= orjson.OPT_INDENT_2
    return orjson.dumps(obj, option=opts, default=default).decode("utf-8")


def loads(s: str | bytes) -> Any:
    """Deserialize JSON string to Python object.

    Args:
        s: JSON string or bytes.

    Returns:
        Deserialized Python object.
    """
    return orjson.loads(s)


def dump(
    obj: Any,
    fp: IO[str],
    *,
    default: Callable[[Any], Any] | None = None,
    indent: bool = False,
    ensure_ascii: bool = True,  # ignored
) -> None:
    """Serialize obj to JSON and write to file.

    Args:
        obj: Object to serialize.
        fp: File-like object to write to.
        default: Function for objects that can't be serialized.
        indent: If True, format with 2-space indentation.
        ensure_ascii: Ignored (kept for stdlib json compatibility).
    """
    fp.write(dumps(obj, default=default, indent=indent))


def load(fp: IO[str] | IO[bytes]) -> Any:
    """Read JSON from file and deserialize.

    Args:
        fp: File-like object to read from.

    Returns:
        Deserialized Python object.
    """
    return loads(fp.read())
