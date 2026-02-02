"""
Utils Export Tests
Tests: Public API exports from utils package

Layer: Utility
Dependencies: utils
"""

from __future__ import annotations


def test_cwe_database_exported():
    """CWEDatabase is exported from vulnflow.utils."""
    from vulnflow.utils import CWEDatabase

    assert CWEDatabase is not None
    assert hasattr(CWEDatabase, "from_xml")


def test_parse_cpe23_exported():
    """parse_cpe23 is exported from vulnflow.utils."""
    from vulnflow.utils import parse_cpe23

    assert callable(parse_cpe23)
