"""Tests for APIEnricher base class."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import numpy as np
import pytest

from vulnflow.dataset import VulnDataset
from vulnflow.registry import FieldDefinition
from vulnflow.transforms.enrichers.api_base import APIEnricher
from vulnflow.utils.http import HttpError

# ---------------------------------------------------------------------------
# Concrete test subclass for APIEnricher
# ---------------------------------------------------------------------------


class _TestAPIEnricher(APIEnricher):
    """Concrete subclass for testing APIEnricher."""

    name: str = "_TestAPIEnricher"
    depends_on: list[str] = []
    OUTPUT_SCHEMA: list[FieldDefinition] = [
        FieldDefinition("test_col", "str", "Test column"),
    ]

    def __init__(
        self,
        fetch_fn: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._fetch_fn = fetch_fn

    async def _fetch_for_cve(self, cve_id: str) -> dict[str, Any] | None:
        if self._fetch_fn is not None:
            return await self._fetch_fn(cve_id)
        return {"cve_id": cve_id, "data": "test"}

    def _apply_results(
        self,
        dataset: VulnDataset,
        results: list[dict[str, Any] | None],
    ) -> None:
        col = np.empty(len(results), dtype=object)
        for i, r in enumerate(results):
            col[i] = json.dumps(r) if r is not None else "null"
        dataset.set_column("test_col", col)


# ---------------------------------------------------------------------------
# APIEnricher base class tests
# ---------------------------------------------------------------------------


class TestAPIEnricherBase:
    """Tests for APIEnricher initialization and configuration."""

    def test_init_defaults(self) -> None:
        enricher = _TestAPIEnricher()
        assert enricher.max_parallel == 10
        assert enricher.rate_limit == 1.0
        assert enricher.max_retries == 3

    def test_init_custom_params(self) -> None:
        enricher = _TestAPIEnricher(max_parallel=5, rate_limit=2.0, max_retries=1)
        assert enricher.max_parallel == 5
        assert enricher.rate_limit == 2.0
        assert enricher.max_retries == 1

    def test_schema_registered(self) -> None:
        """APIEnricher subclass registers schema via BaseTransform.__init__."""
        from vulnflow.registry import Registry

        enricher = _TestAPIEnricher()
        fields = Registry.get_component_columns(enricher.name)
        assert len(fields) == 1
        assert fields[0].name == "test_col"


class TestRetryBehavior:
    """Tests for _fetch_with_retry."""

    @pytest.fixture(autouse=True)
    def _fast_sleep(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Disable asyncio.sleep for fast retry tests."""

        async def _noop_sleep(_seconds: float) -> None:
            pass

        monkeypatch.setattr("asyncio.sleep", _noop_sleep)

    @pytest.mark.asyncio
    async def test_returns_on_first_success(self) -> None:
        call_count = 0

        async def fetch(cve_id: str) -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            return {"cve_id": cve_id}

        enricher = _TestAPIEnricher(fetch_fn=fetch, max_retries=3)
        result = await enricher._fetch_with_retry("CVE-2024-0001")

        assert result is not None
        assert result["cve_id"] == "CVE-2024-0001"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_http_error(self) -> None:
        call_count = 0

        async def fetch(cve_id: str) -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise HttpError("Simulated HTTP failure")
            return {"cve_id": cve_id, "attempt": call_count}

        enricher = _TestAPIEnricher(fetch_fn=fetch, max_retries=3)
        result = await enricher._fetch_with_retry("CVE-2024-0001")

        assert call_count == 3
        assert result is not None
        assert result["attempt"] == 3

    @pytest.mark.asyncio
    async def test_retries_on_os_error(self) -> None:
        """OSError (connection refused, etc.) is retried."""
        call_count = 0

        async def fetch(cve_id: str) -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise OSError("Connection refused")
            return {"cve_id": cve_id}

        enricher = _TestAPIEnricher(fetch_fn=fetch, max_retries=3)
        result = await enricher._fetch_with_retry("CVE-2024-0001")

        assert result is not None
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_returns_none_after_max_retries(self) -> None:
        call_count = 0

        async def fetch(cve_id: str) -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            raise HttpError("Always fails")

        enricher = _TestAPIEnricher(fetch_fn=fetch, max_retries=2)
        result = await enricher._fetch_with_retry("CVE-2024-0001")

        assert result is None
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_non_retryable_errors_propagate(self) -> None:
        """Programming errors (TypeError, KeyError) are NOT retried."""

        async def fetch(cve_id: str) -> dict[str, Any]:
            raise TypeError("Bug in fetch implementation")

        enricher = _TestAPIEnricher(fetch_fn=fetch, max_retries=3)
        with pytest.raises(TypeError, match="Bug in fetch"):
            await enricher._fetch_with_retry("CVE-2024-0001")


class TestConcurrencyLimits:
    """Tests for _fetch_with_limits and semaphore behavior."""

    @pytest.fixture(autouse=True)
    def _fast_sleep(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Disable asyncio.sleep for fast rate-limit tests."""

        async def _noop_sleep(_seconds: float) -> None:
            pass

        monkeypatch.setattr("asyncio.sleep", _noop_sleep)

    @pytest.mark.asyncio
    async def test_concurrency_limited(self) -> None:
        """Verify semaphore limits concurrent requests."""
        max_concurrent = 0
        current_concurrent = 0

        async def fetch(cve_id: str) -> dict[str, Any]:
            nonlocal max_concurrent, current_concurrent
            current_concurrent += 1
            max_concurrent = max(max_concurrent, current_concurrent)
            await asyncio.sleep(0.05)
            current_concurrent -= 1
            return {"cve_id": cve_id}

        enricher = _TestAPIEnricher(
            fetch_fn=fetch,
            max_parallel=2,
            rate_limit=1000.0,  # effectively no rate limit
        )

        cve_ids = [f"CVE-2024-{i:04d}" for i in range(6)]
        await asyncio.gather(
            *(enricher._fetch_with_limits(cve_id) for cve_id in cve_ids)
        )

        assert max_concurrent <= 2


class TestTransform:
    """Tests for the full transform() pipeline."""

    @pytest.mark.asyncio
    async def test_transform_processes_all_cves(self) -> None:
        dataset = VulnDataset()
        dataset.set_column(
            "cve_id",
            np.array(["CVE-2024-0001", "CVE-2024-0002", "CVE-2024-0003"], dtype=object),
        )

        enricher = _TestAPIEnricher(rate_limit=1000.0)
        await enricher.transform(dataset)

        assert "test_col" in dataset.columns
        col = dataset.get_column("test_col")
        assert len(col) == 3
        for val in col:
            parsed = json.loads(val)
            assert "cve_id" in parsed

    @pytest.mark.asyncio
    async def test_transform_handles_failures(self) -> None:
        """Failures produce None results, _apply_results handles them."""

        async def fetch(cve_id: str) -> dict[str, Any] | None:
            if cve_id == "CVE-2024-0002":
                raise HttpError("Simulated failure")
            return {"cve_id": cve_id}

        dataset = VulnDataset()
        dataset.set_column(
            "cve_id",
            np.array(["CVE-2024-0001", "CVE-2024-0002", "CVE-2024-0003"], dtype=object),
        )

        enricher = _TestAPIEnricher(fetch_fn=fetch, max_retries=1, rate_limit=1000.0)
        await enricher.transform(dataset)

        col = dataset.get_column("test_col")
        assert json.loads(col[0])["cve_id"] == "CVE-2024-0001"
        assert col[1] == "null"
        assert json.loads(col[2])["cve_id"] == "CVE-2024-0003"
