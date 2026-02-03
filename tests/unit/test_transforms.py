"""
Transform Protocols - Test protocol definitions
Tests: Enricher, Transform, Inference protocols

Layer: Transform base
Dependencies: None
"""

from __future__ import annotations

import pytest
from vulnflow.transforms.base import Enricher, Transform


def test_enricher_protocol_requires_depends_on():
    """Enricher protocol requires depends_on attribute."""

    class ValidEnricher:
        name = "test"
        depends_on: list[str] = []

        async def transform(self, dataset):
            pass

    # Should satisfy protocol
    assert isinstance(ValidEnricher(), Enricher)


def test_enricher_without_depends_on_fails():
    """Enricher without depends_on doesn't satisfy protocol."""

    class InvalidEnricher:
        name = "test"
        # Missing depends_on

        async def transform(self, dataset):
            pass

    # Should NOT satisfy protocol
    assert not isinstance(InvalidEnricher(), Enricher)
