"""
CWE Loader - Load CWE database from XML
Tests: CWE XML parsing, lookup functions

Layer: Utility
Dependencies: None
"""

from __future__ import annotations

from vulnflow.utils.cwe import CWEDatabase


def test_cwe_database_load_from_xml():
    """CWEDatabase loads from XML file."""
    db = CWEDatabase.from_xml("tests/fixtures/cwe_sample.xml")

    assert len(db) == 4
    assert "CWE-79" in db
    assert "CWE-94" in db
    assert "CWE-502" in db


def test_cwe_database_lookup_name():
    """CWEDatabase.get_name() returns CWE name."""
    db = CWEDatabase.from_xml("tests/fixtures/cwe_sample.xml")

    name = db.get_name("CWE-502")
    assert name == "Deserialization of Untrusted Data"


def test_cwe_database_lookup_category():
    """CWEDatabase.get_category() returns parent CWE ID."""
    db = CWEDatabase.from_xml("tests/fixtures/cwe_sample.xml")

    category = db.get_category("CWE-502")
    assert category == "CWE-913"


def test_cwe_database_lookup_missing():
    """CWEDatabase.get_name() returns None for missing CWE."""
    db = CWEDatabase.from_xml("tests/fixtures/cwe_sample.xml")

    name = db.get_name("CWE-9999")
    assert name is None
