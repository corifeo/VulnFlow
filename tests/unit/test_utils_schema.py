"""Tests for schema introspection utilities."""
from vulnflow.utils.schema import get_flattened_columns, get_record_columns


def test_get_record_columns_returns_list():
    """get_record_columns() returns list of field names."""
    columns = get_record_columns()

    assert isinstance(columns, list)
    assert len(columns) > 0


def test_get_record_columns_contains_id():
    """get_record_columns() includes VulnRecord fields."""
    columns = get_record_columns()

    assert "id" in columns
    assert "cve_id" in columns
    assert "description" in columns
    assert "cvss_scores" in columns


def test_get_flattened_columns_returns_list():
    """get_flattened_columns() returns list of flattened column names."""
    columns = get_flattened_columns()

    assert isinstance(columns, list)
    assert len(columns) > 0


def test_get_flattened_columns_contains_cvss():
    """get_flattened_columns() includes CVSS flattened columns."""
    columns = get_flattened_columns()

    assert "cvss_base_score" in columns
    assert "cvss_base_severity" in columns
    assert "cvss_version" in columns
    assert "cvss_vector_string" in columns
    assert "cvss_source" in columns


def test_get_flattened_columns_contains_cwe():
    """get_flattened_columns() includes CWE flattened columns."""
    columns = get_flattened_columns()

    assert "primary_cwe_id" in columns
    assert "primary_cwe_name" in columns


def test_get_flattened_columns_contains_cpe():
    """get_flattened_columns() includes CPE flattened columns."""
    columns = get_flattened_columns()

    assert "primary_vendor" in columns
    assert "primary_product" in columns
    assert "primary_version" in columns
