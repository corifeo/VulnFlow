"""
Exceptions - Domain-specific error types
Tests: Exception hierarchy, error messages

Layer: Core infrastructure
Dependencies: None
"""

from __future__ import annotations

import pytest

from vulnflow.exceptions import (
    EnricherError,
    ExtractorError,
    LoaderError,
    TransformError,
    VulnFlowError,
)


def test_exception_hierarchy():
    """Domain exceptions inherit from VulnFlowError."""
    assert issubclass(ExtractorError, VulnFlowError)
    assert issubclass(EnricherError, VulnFlowError)
    assert issubclass(TransformError, VulnFlowError)
    assert issubclass(LoaderError, VulnFlowError)


def test_exceptions_are_raisable():
    """Domain exceptions can be raised with messages."""
    with pytest.raises(EnricherError, match="test error"):
        raise EnricherError("test error")

    with pytest.raises(TransformError, match="transform failed"):
        raise TransformError("transform failed")


def test_extractor_error_creation():
    """ExtractorError preserves error message."""
    msg = "Failed to parse NVD JSON: invalid format"
    err = ExtractorError(msg)
    assert str(err) == msg
    assert isinstance(err, VulnFlowError)
    assert isinstance(err, Exception)


def test_loader_error_creation():
    """LoaderError preserves error message."""
    msg = "Failed to write Parquet file: permission denied"
    err = LoaderError(msg)
    assert str(err) == msg
    assert isinstance(err, VulnFlowError)
    assert isinstance(err, Exception)


def test_extractor_error_raisable():
    """ExtractorError can be raised and caught."""
    with pytest.raises(ExtractorError, match="extraction failed"):
        raise ExtractorError("extraction failed")


def test_loader_error_raisable():
    """LoaderError can be raised and caught."""
    with pytest.raises(LoaderError, match="load failed"):
        raise LoaderError("load failed")


def test_extractor_error_catchable_as_base():
    """ExtractorError can be caught as VulnFlowError."""
    try:
        raise ExtractorError("test")
    except VulnFlowError:
        pass  # Expected
    else:
        pytest.fail("ExtractorError should be catchable as VulnFlowError")


def test_loader_error_catchable_as_base():
    """LoaderError can be caught as VulnFlowError."""
    try:
        raise LoaderError("test")
    except VulnFlowError:
        pass  # Expected
    else:
        pytest.fail("LoaderError should be catchable as VulnFlowError")
