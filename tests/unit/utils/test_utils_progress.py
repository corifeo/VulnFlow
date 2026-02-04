"""
Progress bar utilities - tqdm wrapper for Jupyter/terminal
Tests: progress_bar() factory function
"""

from __future__ import annotations

from vulnflow.utils.progress import progress_bar


def test_progress_bar_creates_tqdm_instance():
    """progress_bar() returns tqdm instance."""
    pbar = progress_bar(total=10, desc="Test")

    assert pbar is not None
    assert pbar.total == 10
    assert pbar.desc == "Test"

    pbar.close()


def test_progress_bar_applies_defaults():
    """progress_bar() applies VulnFlow defaults."""
    pbar = progress_bar(total=100)

    assert pbar.unit == "records"
    assert pbar.unit_scale is True
    assert pbar.leave is True

    pbar.close()


def test_progress_bar_allows_override():
    """progress_bar() allows overriding defaults."""
    pbar = progress_bar(total=50, unit="items", leave=False)

    assert pbar.unit == "items"
    assert pbar.leave is False

    pbar.close()
