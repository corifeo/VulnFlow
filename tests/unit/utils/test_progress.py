"""Tests for progress bar utilities — ProgressReporter and suppression.

Layer: Utils
Dependencies: progress
"""

from __future__ import annotations

import os
import time
from unittest.mock import MagicMock, patch

import pytest

from vulnflow.utils.progress import (
    ProgressMode,
    ProgressReporter,
    _detect_mode,
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


def test_progress_bar_returns_suppressed_when_tqdm_disable_set() -> None:
    """progress_bar() returns _SuppressedProgress when TQDM_DISABLE=1."""
    os.environ["TQDM_DISABLE"] = "1"
    try:
        bar = progress_bar(total=100, desc="Test")
        assert isinstance(bar, _SuppressedProgress)
    finally:
        os.environ.pop("TQDM_DISABLE", None)


def test_progress_bar_returns_tqdm_when_not_suppressed() -> None:
    """progress_bar() returns real tqdm when not suppressed."""
    os.environ.pop("TQDM_DISABLE", None)
    bar = progress_bar(total=10, desc="Test")
    try:
        assert not isinstance(bar, _SuppressedProgress)
    finally:
        bar.close()


# =============================================================================
# ProgressReporter — LOG Mode Tests
# =============================================================================


def test_reporter_log_mode_stage_lifecycle(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """ProgressReporter prints checkmark line on stage completion in LOG mode."""
    os.environ["VULNFLOW_PROGRESS_MODE"] = "log"
    try:
        with ProgressReporter("Testing", total=2, unit="items") as rpt:
            rpt.stage_started("StageA")
            rpt.stage_completed("StageA", detail="100 records")

            rpt.stage_started("StageB")
            rpt.stage_completed("StageB")
    finally:
        os.environ.pop("VULNFLOW_PROGRESS_MODE", None)

    captured = capsys.readouterr()
    assert "+ StageA (100 records," in captured.out
    assert "+ StageB (" in captured.out
    assert "100 records" not in captured.out.split("StageB")[1]


def test_reporter_log_mode_stage_failed(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """ProgressReporter prints cross mark line on stage failure in LOG mode."""
    os.environ["VULNFLOW_PROGRESS_MODE"] = "log"
    try:
        with ProgressReporter("Testing", total=1, unit="items") as rpt:
            rpt.stage_started("BadStage")
            rpt.stage_failed("BadStage", error="timeout after 30s")
    finally:
        os.environ.pop("VULNFLOW_PROGRESS_MODE", None)

    captured = capsys.readouterr()
    assert "x BadStage" in captured.out
    assert "timeout after 30s" in captured.out


def test_reporter_log_mode_summary_line(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """ProgressReporter.summary() prints arrow line in LOG mode."""
    os.environ["VULNFLOW_PROGRESS_MODE"] = "log"
    try:
        with ProgressReporter("Testing", total=1, unit="items") as rpt:
            rpt.stage_started("S1")
            rpt.stage_completed("S1")
            rpt.summary("55,034 CVEs total")
    finally:
        os.environ.pop("VULNFLOW_PROGRESS_MODE", None)

    captured = capsys.readouterr()
    assert "> 55,034 CVEs total" in captured.out


def test_reporter_log_mode_suppresses_inner_progress_bars() -> None:
    """Inner progress_bar() calls return _SuppressedProgress in LOG mode."""
    os.environ.pop("TQDM_DISABLE", None)
    os.environ["VULNFLOW_PROGRESS_MODE"] = "log"
    try:
        with ProgressReporter("Testing", total=1, unit="items"):
            inner = progress_bar(total=100, desc="Inner")
            assert isinstance(inner, _SuppressedProgress)

        # After exit, suppression is cleared
        outer = progress_bar(total=10, desc="Outer")
        assert not isinstance(outer, _SuppressedProgress)
        outer.close()
    finally:
        os.environ.pop("TQDM_DISABLE", None)
        os.environ.pop("VULNFLOW_PROGRESS_MODE", None)


def test_reporter_tracks_duration() -> None:
    """stage_completed duration reflects actual elapsed time."""
    os.environ["VULNFLOW_PROGRESS_MODE"] = "log"
    try:
        with ProgressReporter("Testing", total=1, unit="items") as rpt:
            rpt.stage_started("Slow")
            time.sleep(0.05)
            rpt.stage_completed("Slow")
    finally:
        os.environ.pop("VULNFLOW_PROGRESS_MODE", None)


# =============================================================================
# ProgressReporter — QUIET Mode (Dot Progress) Tests
# =============================================================================


def test_reporter_quiet_mode_prints_dots(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """ProgressReporter prints '.' per stage in QUIET mode."""
    os.environ["VULNFLOW_PROGRESS_MODE"] = "quiet"
    try:
        with ProgressReporter("Testing", total=3, unit="items") as rpt:
            rpt.stage_started("A")
            rpt.stage_completed("A", detail="100 records")
            rpt.stage_started("B")
            rpt.stage_completed("B")
            rpt.stage_started("C")
            rpt.stage_completed("C")
    finally:
        os.environ.pop("VULNFLOW_PROGRESS_MODE", None)

    captured = capsys.readouterr()
    assert captured.out == "..."


def test_reporter_quiet_mode_prints_bang_on_failure(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """ProgressReporter prints '!' for failed stages in quiet mode."""
    os.environ["VULNFLOW_PROGRESS_MODE"] = "quiet"
    try:
        with ProgressReporter("Testing", total=3, unit="items") as rpt:
            rpt.stage_started("A")
            rpt.stage_completed("A")
            rpt.stage_started("B")
            rpt.stage_failed("B", error="timeout")
            rpt.stage_started("C")
            rpt.stage_completed("C")
    finally:
        os.environ.pop("VULNFLOW_PROGRESS_MODE", None)

    captured = capsys.readouterr()
    assert captured.out == ".!."


def test_reporter_quiet_mode_suppresses_summary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """ProgressReporter.summary() is suppressed in quiet mode."""
    os.environ["VULNFLOW_PROGRESS_MODE"] = "quiet"
    try:
        with ProgressReporter("Testing", total=1, unit="items") as rpt:
            rpt.stage_started("A")
            rpt.stage_completed("A")
            rpt.summary("55,034 CVEs total")
    finally:
        os.environ.pop("VULNFLOW_PROGRESS_MODE", None)

    captured = capsys.readouterr()
    assert "CVEs" not in captured.out
    assert captured.out == "."


def test_reporter_quiet_mode_no_unicode(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Quiet mode output contains no Unicode — safe for cp1252 consoles."""
    os.environ["VULNFLOW_PROGRESS_MODE"] = "quiet"
    try:
        with ProgressReporter("Testing", total=2, unit="items") as rpt:
            rpt.stage_started("A")
            rpt.stage_completed("A")
            rpt.stage_started("B")
            rpt.stage_failed("B", error="crash")
    finally:
        os.environ.pop("VULNFLOW_PROGRESS_MODE", None)

    captured = capsys.readouterr()
    assert captured.out.isascii()


# =============================================================================
# ProgressReporter — SILENT Mode Tests
# =============================================================================


def test_reporter_silent_mode_no_output(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """ProgressReporter produces zero output in SILENT mode (_suppress_progress)."""
    token = _suppress_progress.set(True)
    try:
        with ProgressReporter("Testing", total=3, unit="items") as rpt:
            rpt.stage_started("A")
            rpt.stage_completed("A", detail="100 records")
            rpt.stage_started("B")
            rpt.stage_failed("B", error="timeout")
            rpt.stage_started("C")
            rpt.stage_completed("C")
            rpt.summary("done")
    finally:
        _suppress_progress.reset(token)

    captured = capsys.readouterr()
    assert captured.out == ""


# =============================================================================
# ProgressReporter — BAR Mode Tests
# =============================================================================


def test_reporter_bar_mode_no_text_output(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """BAR mode produces no + glyph text output even with detail."""
    os.environ["VULNFLOW_PROGRESS_MODE"] = "bar"
    os.environ.pop("TQDM_DISABLE", None)
    try:
        with ProgressReporter("Testing", total=2, unit="items") as rpt:
            rpt.stage_started("A")
            rpt.stage_completed("A", detail="100 records")
            rpt.stage_started("B")
            rpt.stage_completed("B")
    finally:
        os.environ.pop("VULNFLOW_PROGRESS_MODE", None)

    captured = capsys.readouterr()
    assert "+" not in captured.out


def test_reporter_bar_mode_sets_postfix_on_detail() -> None:
    """BAR mode sets postfix with compact format when detail is non-empty."""
    os.environ["VULNFLOW_PROGRESS_MODE"] = "bar"
    os.environ.pop("TQDM_DISABLE", None)
    try:
        with ProgressReporter("Testing", total=1, unit="items") as rpt:
            mock_bar = MagicMock()
            rpt._bar = mock_bar

            rpt.stage_started("A")
            rpt.stage_completed("A", detail="100 records", duration=1.5)

        mock_bar.set_postfix_str.assert_any_call("100 records in 1.5s")
        mock_bar.update.assert_called_with(1)
    finally:
        os.environ.pop("VULNFLOW_PROGRESS_MODE", None)


def test_reporter_bar_mode_updates_bar() -> None:
    """BAR mode calls bar.update() and set_postfix_str() on completion."""
    os.environ["VULNFLOW_PROGRESS_MODE"] = "bar"
    os.environ.pop("TQDM_DISABLE", None)
    try:
        with ProgressReporter("Testing", total=2, unit="items") as rpt:
            mock_bar = MagicMock()
            rpt._bar = mock_bar

            rpt.stage_started("A")
            rpt.stage_completed("A", detail="100 records", duration=2.0)

        mock_bar.update.assert_called_with(1)
        mock_bar.set_postfix_str.assert_any_call("100 records in 2.0s")
    finally:
        os.environ.pop("VULNFLOW_PROGRESS_MODE", None)


def test_reporter_bar_mode_failure_writes_error() -> None:
    """BAR mode writes failure text via tqdm.write() so it appears above the bar."""
    os.environ["VULNFLOW_PROGRESS_MODE"] = "bar"
    os.environ.pop("TQDM_DISABLE", None)
    try:
        with patch("vulnflow.utils.progress._tqdm.write") as mock_write:
            with ProgressReporter("Testing", total=1, unit="items") as rpt:
                mock_bar = MagicMock()
                rpt._bar = mock_bar

                rpt.stage_started("Bad")
                rpt.stage_failed("Bad", error="timeout")

            # tqdm.write called for failure + summary is separate
            written = mock_write.call_args_list[0][0][0]
            assert "x Bad" in written
            assert "timeout" in written
            mock_bar.update.assert_called_with(1)
    finally:
        os.environ.pop("VULNFLOW_PROGRESS_MODE", None)


def test_reporter_bar_mode_summary_sets_postfix() -> None:
    """BAR mode summary sets postfix on the bar."""
    os.environ["VULNFLOW_PROGRESS_MODE"] = "bar"
    os.environ.pop("TQDM_DISABLE", None)
    try:
        with ProgressReporter("Testing", total=1, unit="items") as rpt:
            mock_bar = MagicMock()
            rpt._bar = mock_bar

            rpt.stage_started("A")
            rpt.stage_completed("A")
            rpt.summary("55,034 CVEs total")

        mock_bar.set_postfix_str.assert_any_call("55,034 CVEs total")
    finally:
        os.environ.pop("VULNFLOW_PROGRESS_MODE", None)


def test_reporter_bar_mode_started_sets_postfix() -> None:
    """BAR mode stage_started sets postfix to stage name."""
    os.environ["VULNFLOW_PROGRESS_MODE"] = "bar"
    os.environ.pop("TQDM_DISABLE", None)
    try:
        with ProgressReporter("Testing", total=1, unit="items") as rpt:
            mock_bar = MagicMock()
            rpt._bar = mock_bar

            rpt.stage_started("A")

        mock_bar.set_postfix_str.assert_any_call("A")
    finally:
        os.environ.pop("VULNFLOW_PROGRESS_MODE", None)


def test_reporter_bar_mode_does_not_suppress_inner_bars() -> None:
    """BAR mode lets inner progress_bar() calls create real bars."""
    os.environ["VULNFLOW_PROGRESS_MODE"] = "bar"
    os.environ.pop("TQDM_DISABLE", None)
    try:
        with ProgressReporter("Testing", total=1, unit="items"):
            inner = progress_bar(total=100, desc="Inner")
            assert not isinstance(inner, _SuppressedProgress)
            inner.close()
    finally:
        os.environ.pop("VULNFLOW_PROGRESS_MODE", None)


def test_reporter_bar_format_parameter() -> None:
    """ProgressReporter passes bar_format to the outer bar."""
    os.environ["VULNFLOW_PROGRESS_MODE"] = "bar"
    os.environ.pop("TQDM_DISABLE", None)
    try:
        fmt = "{desc}: {n_fmt}/{total_fmt} [{elapsed}]"
        with ProgressReporter(
            "Testing", total=1, unit="items", bar_format=fmt
        ) as rpt:
            assert rpt._bar is not None
            assert rpt._bar.bar_format == fmt
    finally:
        os.environ.pop("VULNFLOW_PROGRESS_MODE", None)


# =============================================================================
# Mode Detection Tests
# =============================================================================


def test_detect_mode_returns_silent_when_suppressed() -> None:
    """_detect_mode() returns SILENT when _suppress_progress ContextVar is True."""
    token = _suppress_progress.set(True)
    try:
        assert _detect_mode() is ProgressMode.SILENT
    finally:
        _suppress_progress.reset(token)


def test_detect_mode_respects_env_var() -> None:
    """_detect_mode() uses VULNFLOW_PROGRESS_MODE env var when set."""
    for mode_str, expected in [
        ("quiet", ProgressMode.QUIET),
        ("bar", ProgressMode.BAR),
        ("log", ProgressMode.LOG),
        ("BAR", ProgressMode.BAR),  # Case insensitive
    ]:
        os.environ["VULNFLOW_PROGRESS_MODE"] = mode_str
        try:
            assert _detect_mode() is expected, f"Failed for {mode_str!r}"
        finally:
            os.environ.pop("VULNFLOW_PROGRESS_MODE", None)


def test_detect_mode_returns_log_for_non_tty() -> None:
    """_detect_mode() returns LOG when stdout is not a tty and not a notebook."""
    os.environ.pop("VULNFLOW_PROGRESS_MODE", None)
    with patch("vulnflow.utils.progress._is_notebook", return_value=False):
        # pytest stdout is not a tty
        assert _detect_mode() is ProgressMode.LOG


def test_detect_mode_returns_bar_for_tty() -> None:
    """_detect_mode() returns BAR when stdout is a tty."""
    os.environ.pop("VULNFLOW_PROGRESS_MODE", None)
    with (
        patch("vulnflow.utils.progress._is_notebook", return_value=False),
        patch("vulnflow.utils.progress.sys.stdout") as mock_stdout,
    ):
        mock_stdout.isatty.return_value = True
        assert _detect_mode() is ProgressMode.BAR


def test_detect_mode_returns_bar_for_notebook() -> None:
    """_detect_mode() returns BAR when in a Jupyter notebook."""
    os.environ.pop("VULNFLOW_PROGRESS_MODE", None)
    with patch("vulnflow.utils.progress._is_notebook", return_value=True):
        assert _detect_mode() is ProgressMode.BAR


# =============================================================================
# Inner Suppression Robustness Tests
# =============================================================================


def test_reporter_log_sets_tqdm_disable_for_inner_suppression() -> None:
    """LOG mode sets TQDM_DISABLE=1 inside context for belt-and-suspenders."""
    os.environ.pop("TQDM_DISABLE", None)
    os.environ["VULNFLOW_PROGRESS_MODE"] = "log"
    try:
        with ProgressReporter("Testing", total=1, unit="items"):
            assert os.environ.get("TQDM_DISABLE") == "1"

        # After exit, TQDM_DISABLE should be restored (was not set before)
        assert "TQDM_DISABLE" not in os.environ
    finally:
        os.environ.pop("VULNFLOW_PROGRESS_MODE", None)
        os.environ.pop("TQDM_DISABLE", None)


def test_reporter_log_restores_tqdm_disable_on_exit() -> None:
    """LOG mode restores previous TQDM_DISABLE value on exit."""
    os.environ["TQDM_DISABLE"] = "0"
    os.environ["VULNFLOW_PROGRESS_MODE"] = "log"
    try:
        with ProgressReporter("Testing", total=1, unit="items"):
            assert os.environ.get("TQDM_DISABLE") == "1"

        assert os.environ.get("TQDM_DISABLE") == "0"
    finally:
        os.environ.pop("TQDM_DISABLE", None)
        os.environ.pop("VULNFLOW_PROGRESS_MODE", None)


def test_reporter_bar_mode_does_not_set_tqdm_disable() -> None:
    """BAR mode does not set TQDM_DISABLE (inner bars should show)."""
    os.environ.pop("TQDM_DISABLE", None)
    os.environ["VULNFLOW_PROGRESS_MODE"] = "bar"
    try:
        with ProgressReporter("Testing", total=1, unit="items"):
            assert os.environ.get("TQDM_DISABLE") is None

        assert os.environ.get("TQDM_DISABLE") is None
    finally:
        os.environ.pop("VULNFLOW_PROGRESS_MODE", None)
        os.environ.pop("TQDM_DISABLE", None)


def test_quiet_mode_suppresses_via_tqdm_disable_in_threads() -> None:
    """progress_bar() returns _SuppressedProgress when TQDM_DISABLE=1.

    This handles threads that don't inherit the _suppress_progress ContextVar.
    """
    os.environ["TQDM_DISABLE"] = "1"
    try:
        # Even without ContextVar, env var suppresses
        bar = progress_bar(total=100, desc="Thread inner")
        assert isinstance(bar, _SuppressedProgress)
    finally:
        os.environ.pop("TQDM_DISABLE", None)


# =============================================================================
# ProgressReporter — Postfix Clearing Tests
# =============================================================================


def test_reporter_bar_exit_clears_postfix() -> None:
    """BAR mode __exit__ clears postfix before closing bar."""
    os.environ["VULNFLOW_PROGRESS_MODE"] = "bar"
    os.environ.pop("TQDM_DISABLE", None)
    try:
        with ProgressReporter("Testing", total=1, unit="items") as rpt:
            mock_bar = MagicMock()
            rpt._bar = mock_bar

            rpt.stage_started("A")
            rpt.stage_completed("A", detail="100 records", duration=1.0)

        # __exit__ should clear postfix then close
        calls = [c[0] for c in mock_bar.set_postfix_str.call_args_list]
        # Last set_postfix_str call should be ("",) to clear stale postfix
        assert calls[-1] == ("",)
        mock_bar.close.assert_called_once()
    finally:
        os.environ.pop("VULNFLOW_PROGRESS_MODE", None)
