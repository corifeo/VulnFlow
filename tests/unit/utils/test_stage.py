"""StageObservability mixin tests."""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from vulnflow.utils.stage import StageObservability


class FakeStage(StageObservability):
    """Minimal stage for testing."""

    name = "fake"
    label = "fake-label"

    def __init__(self) -> None:
        self._init_observability()


class TestInitObservability:
    def test_sets_logger_with_module_name(self) -> None:
        stage = FakeStage()
        assert isinstance(stage.logger, logging.Logger)
        assert stage.logger.name == FakeStage.__module__

    def test_sets_empty_timings(self) -> None:
        stage = FakeStage()
        assert stage._timings == {}

    def test_idempotent(self) -> None:
        stage = FakeStage()
        stage._timings["old"] = 1.0
        stage._init_observability()
        assert stage._timings == {}


class TestTimed:
    def test_records_duration(self) -> None:
        stage = FakeStage()
        with stage._timed("work"):
            pass
        assert "work" in stage._timings
        assert stage._timings["work"] >= 0.0

    def test_multiple_phases(self) -> None:
        stage = FakeStage()
        with stage._timed("a"):
            pass
        with stage._timed("b"):
            pass
        assert "a" in stage._timings
        assert "b" in stage._timings

    def test_records_timing_on_exception(self) -> None:
        stage = FakeStage()
        with pytest.raises(ValueError):
            with stage._timed("fail"):
                raise ValueError("boom")
        assert "fail" in stage._timings
        assert stage._timings["fail"] >= 0.0


class TestProgress:
    def test_uses_label_as_default_desc(self) -> None:
        stage = FakeStage()
        with patch("vulnflow.utils.stage.progress_bar") as mock_pb:
            stage._progress(total=10, unit="items")
            mock_pb.assert_called_once_with(total=10, desc="fake-label", unit="items")

    def test_desc_override(self) -> None:
        stage = FakeStage()
        with patch("vulnflow.utils.stage.progress_bar") as mock_pb:
            stage._progress(total=5, desc="custom", unit="months")
            mock_pb.assert_called_once_with(total=5, desc="custom", unit="months")

    def test_falls_back_to_name(self) -> None:
        class NoLabel(StageObservability):
            name = "no-label"

            def __init__(self) -> None:
                self._init_observability()

        stage = NoLabel()
        with patch("vulnflow.utils.stage.progress_bar") as mock_pb:
            stage._progress(total=1)
            mock_pb.assert_called_once_with(total=1, desc="no-label", unit="items")
