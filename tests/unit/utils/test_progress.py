"""Tests for progress bar utilities — ProgressReporter and suppression.

Layer: Utils
Dependencies: progress
"""

from __future__ import annotations

import os
import time

import pytest

from vulnflow.utils.progress import (
    ProgressReporter,
    _suppress_progress,
    _SuppressedProgress,
    progress_bar,
)

# =============================================================================
# _SuppressedProgress Tests
# =============================================================================


def test_suppressed_progress_update_is_silent() -> None:
    """_SuppressedProgress.update() does nothing and doesn't raise."""
    noop = _SuppressedProgress()
    noop.update(5)
    noop.update(1)


def test_suppressed_progress_context_manager() -> None:
    """_SuppressedProgress works as a context manager."""
    with _SuppressedProgress() as p:
        p.update(1)
        p.set_postfix_str("test")
        p.close()


# =============================================================================
# Suppression ContextVar Tests
# =============================================================================


def test_progress_bar_returns_suppressed_when_flag_set() -> None:
    """progress_bar() returns _SuppressedProgress when _suppress_progress is True."""
    token = _suppress_progress.set(True)
    try:
        bar = progress_bar(total=100, desc="Test")
        assert isinstance(bar, _SuppressedProgress)
    finally:
        _suppress_progress.reset(token)


def test_progress_bar_returns_tqdm_when_not_suppressed() -> None:
    """progress_bar() returns real tqdm when _suppress_progress is default."""
    os.environ["TQDM_DISABLE"] = "1"
    try:
        bar = progress_bar(total=10, desc="Test")
        assert not isinstance(bar, _SuppressedProgress)
        bar.close()
    finally:
        os.environ.pop("TQDM_DISABLE", None)


# =============================================================================
# ProgressReporter Tests
# =============================================================================


def test_reporter_stage_lifecycle(capsys: pytest.CaptureFixture[str]) -> None:
    """ProgressReporter prints checkmark line on stage completion with duration."""
    os.environ["TQDM_DISABLE"] = "1"
    try:
        with ProgressReporter("Testing", total=2, unit="items") as rpt:
            rpt.stage_started("StageA")
            rpt.stage_completed("StageA", detail="100 records")

            rpt.stage_started("StageB")
            rpt.stage_completed("StageB")
    finally:
        os.environ.pop("TQDM_DISABLE", None)

    captured = capsys.readouterr()
    assert "\u2713 StageA (100 records," in captured.out
    assert "\u2713 StageB (" in captured.out
    assert "100 records" not in captured.out.split("StageB")[1]


def test_reporter_stage_failed(capsys: pytest.CaptureFixture[str]) -> None:
    """ProgressReporter prints cross mark line on stage failure."""
    os.environ["TQDM_DISABLE"] = "1"
    try:
        with ProgressReporter("Testing", total=1, unit="items") as rpt:
            rpt.stage_started("BadStage")
            rpt.stage_failed("BadStage", error="timeout after 30s")
    finally:
        os.environ.pop("TQDM_DISABLE", None)

    captured = capsys.readouterr()
    assert "\u2717 BadStage" in captured.out
    assert "timeout after 30s" in captured.out


def test_reporter_summary_line(capsys: pytest.CaptureFixture[str]) -> None:
    """ProgressReporter.summary() prints arrow line without ticking bar."""
    os.environ["TQDM_DISABLE"] = "1"
    try:
        with ProgressReporter("Testing", total=1, unit="items") as rpt:
            rpt.stage_started("S1")
            rpt.stage_completed("S1")
            rpt.summary("55,034 CVEs total")
    finally:
        os.environ.pop("TQDM_DISABLE", None)

    captured = capsys.readouterr()
    assert "\u2192 55,034 CVEs total" in captured.out


def test_reporter_suppresses_inner_progress_bars() -> None:
    """Inner progress_bar() calls return _SuppressedProgress inside ProgressReporter."""
    os.environ["TQDM_DISABLE"] = "1"
    try:
        with ProgressReporter("Testing", total=1, unit="items"):
            inner = progress_bar(total=100, desc="Inner")
            assert isinstance(inner, _SuppressedProgress)

        outer = progress_bar(total=10, desc="Outer")
        assert not isinstance(outer, _SuppressedProgress)
        outer.close()
    finally:
        os.environ.pop("TQDM_DISABLE", None)


def test_reporter_tracks_duration() -> None:
    """stage_completed duration reflects actual elapsed time."""
    os.environ["TQDM_DISABLE"] = "1"
    try:
        with ProgressReporter("Testing", total=1, unit="items") as rpt:
            rpt.stage_started("Slow")
            time.sleep(0.05)
            rpt.stage_completed("Slow")
    finally:
        os.environ.pop("TQDM_DISABLE", None)


# =============================================================================
# Quiet Mode (Dot Progress) Tests
# =============================================================================


def test_reporter_quiet_mode_prints_dots(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """ProgressReporter prints '.' per stage when _suppress_progress is set."""
    token = _suppress_progress.set(True)
    try:
        with ProgressReporter("Testing", total=3, unit="items") as rpt:
            rpt.stage_started("A")
            rpt.stage_completed("A", detail="100 records")
            rpt.stage_started("B")
            rpt.stage_completed("B")
            rpt.stage_started("C")
            rpt.stage_completed("C")
    finally:
        _suppress_progress.reset(token)

    captured = capsys.readouterr()
    assert captured.out == "...\n"


def test_reporter_quiet_mode_prints_bang_on_failure(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """ProgressReporter prints '!' for failed stages in quiet mode."""
    token = _suppress_progress.set(True)
    try:
        with ProgressReporter("Testing", total=3, unit="items") as rpt:
            rpt.stage_started("A")
            rpt.stage_completed("A")
            rpt.stage_started("B")
            rpt.stage_failed("B", error="timeout")
            rpt.stage_started("C")
            rpt.stage_completed("C")
    finally:
        _suppress_progress.reset(token)

    captured = capsys.readouterr()
    assert captured.out == ".!.\n"


def test_reporter_quiet_mode_suppresses_summary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """ProgressReporter.summary() is suppressed in quiet mode."""
    token = _suppress_progress.set(True)
    try:
        with ProgressReporter("Testing", total=1, unit="items") as rpt:
            rpt.stage_started("A")
            rpt.stage_completed("A")
            rpt.summary("55,034 CVEs total")
    finally:
        _suppress_progress.reset(token)

    captured = capsys.readouterr()
    assert "CVEs" not in captured.out
    assert captured.out == ".\n"


def test_reporter_quiet_mode_no_unicode(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Quiet mode output contains no Unicode — safe for cp1252 consoles."""
    token = _suppress_progress.set(True)
    try:
        with ProgressReporter("Testing", total=2, unit="items") as rpt:
            rpt.stage_started("A")
            rpt.stage_completed("A")
            rpt.stage_started("B")
            rpt.stage_failed("B", error="crash")
    finally:
        _suppress_progress.reset(token)

    captured = capsys.readouterr()
    assert captured.out.isascii()
