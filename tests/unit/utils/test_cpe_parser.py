"""
CPE Parser - Parse CPE 2.3 formatted strings
Tests: Basic parsing, wildcards, invalid formats

Layer: Utility
Dependencies: schema
"""

from __future__ import annotations

import pytest

from vulnflow.utils.cpe import parse_cpe23


def test_parse_cpe23_basic():
    """Parse basic CPE 2.3 string."""
    cpe_str = "cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*"
    cpe = parse_cpe23(cpe_str)

    assert cpe.vendor == "apache"
    assert cpe.product == "log4j"
    assert cpe.version == "2.14.1"
    assert cpe.raw == cpe_str


def test_parse_cpe23_with_wildcard_version():
    """Parse CPE with wildcard version."""
    cpe_str = "cpe:2.3:a:microsoft:windows:*:*:*:*:*:*:*:*"
    cpe = parse_cpe23(cpe_str)

    assert cpe.vendor == "microsoft"
    assert cpe.product == "windows"
    assert cpe.version == "*"


def test_parse_cpe23_invalid_format():
    """Parse invalid CPE raises ValueError."""
    with pytest.raises(ValueError, match="Invalid CPE 2.3 format"):
        parse_cpe23("not-a-cpe")


def test_parse_cpe23_empty():
    """Parse empty CPE raises ValueError."""
    with pytest.raises(ValueError, match="CPE string cannot be empty"):
        parse_cpe23("")
