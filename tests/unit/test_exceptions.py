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
