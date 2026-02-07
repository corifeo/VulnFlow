"""Tests for APIEnricher base class and ExploitDBEnricher."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import numpy as np
import pytest

from vulnflow.dataset import VulnDataset
from vulnflow.registry import FieldDefinition
from vulnflow.transforms.enrichers.api_base import APIEnricher
from vulnflow.transforms.enrichers.exploit import ExploitDBEnricher
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
            np.array(
                ["CVE-2024-0001", "CVE-2024-0002", "CVE-2024-0003"], dtype=object
            ),
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
            np.array(
                ["CVE-2024-0001", "CVE-2024-0002", "CVE-2024-0003"], dtype=object
            ),
        )

        enricher = _TestAPIEnricher(fetch_fn=fetch, max_retries=1, rate_limit=1000.0)
        await enricher.transform(dataset)

        col = dataset.get_column("test_col")
        assert json.loads(col[0])["cve_id"] == "CVE-2024-0001"
        assert col[1] == "null"
        assert json.loads(col[2])["cve_id"] == "CVE-2024-0003"


# ---------------------------------------------------------------------------
# ExploitDBEnricher tests
# ---------------------------------------------------------------------------


class TestExploitDBEnricher:
    """Tests for ExploitDBEnricher."""

    def test_init_defaults(self) -> None:
        enricher = ExploitDBEnricher()
        assert enricher.name == "ExploitDBEnricher"
        assert enricher.max_parallel == 5
        assert enricher.rate_limit == 2.0
        assert enricher.max_retries == 3

    def test_registered_as_non_default(self) -> None:
        """ExploitDBEnricher registered with default=False."""
        from vulnflow.registry import Registry

        defaults = Registry.get_default_transforms()
        default_names = [inst.name for inst in defaults]
        assert "ExploitDBEnricher" not in default_names

    def test_schema_registered(self) -> None:
        from vulnflow.registry import Registry

        enricher = ExploitDBEnricher()
        fields = Registry.get_component_columns(enricher.name)
        assert len(fields) >= 1
        field_names = [f.name for f in fields]
        assert "exploits" in field_names

    @pytest.mark.asyncio
    async def test_apply_results_creates_column(self) -> None:
        """_apply_results creates exploits column when not present."""
        dataset = VulnDataset()
        dataset.set_column(
            "cve_id", np.array(["CVE-2024-0001", "CVE-2024-0002"], dtype=object)
        )

        enricher = ExploitDBEnricher()
        results: list[dict[str, Any] | None] = [
            {
                "source": "exploitdb",
                "id": "12345",
                "url": "https://exploit-db.com/12345",
                "description": "Test exploit",
                "published_at": None,
                "confidence": 1.0,
            },
            None,
        ]

        enricher._apply_results(dataset, results)

        exploits = dataset.get_column("exploits")
        assert len(exploits) == 2

        # First CVE has exploit
        parsed = json.loads(exploits[0])
        assert isinstance(parsed, list)
        assert len(parsed) == 1
        assert parsed[0]["source"] == "exploitdb"
        assert parsed[0]["confidence"] == 1.0

        # Second CVE has empty array
        assert json.loads(exploits[1]) == []

    @pytest.mark.asyncio
    async def test_apply_results_appends_to_existing(self) -> None:
        """_apply_results appends to existing exploits column."""
        dataset = VulnDataset()
        dataset.set_column(
            "cve_id", np.array(["CVE-2024-0001"], dtype=object)
        )
        # Pre-populate with an existing exploit
        existing = [{"source": "github", "id": "gh-001", "confidence": 0.5}]
        dataset.set_column(
            "exploits", np.array([json.dumps(existing)], dtype=object)
        )

        enricher = ExploitDBEnricher()
        results: list[dict[str, Any] | None] = [
            {
                "source": "exploitdb",
                "id": "edb-999",
                "url": "https://exploit-db.com/999",
                "description": "New exploit",
                "published_at": None,
                "confidence": 1.0,
            },
        ]

        enricher._apply_results(dataset, results)

        exploits = json.loads(dataset.get_column("exploits")[0])
        assert len(exploits) == 2
        assert exploits[0]["source"] == "github"
        assert exploits[1]["source"] == "exploitdb"

    @pytest.mark.asyncio
    async def test_transform_with_mock_fetch(self) -> None:
        """Full transform with mocked _fetch_for_cve."""
        dataset = VulnDataset()
        dataset.set_column(
            "cve_id", np.array(["CVE-2024-0001", "CVE-2024-0002"], dtype=object)
        )

        enricher = ExploitDBEnricher()

        async def mock_fetch(cve_id: str) -> dict[str, Any] | None:
            if cve_id == "CVE-2024-0001":
                return {
                    "source": "exploitdb",
                    "id": "99999",
                    "url": "https://exploit-db.com/99999",
                    "description": "Mock exploit",
                    "published_at": None,
                    "confidence": 1.0,
                }
            return None

        enricher._fetch_for_cve = mock_fetch  # type: ignore[assignment]

        await enricher.transform(dataset)

        exploits = dataset.get_column("exploits")
        first = json.loads(exploits[0])
        assert first[0]["id"] == "99999"
        second = json.loads(exploits[1])
        assert second == []

    @pytest.mark.asyncio
    async def test_exploitdb_placeholder_returns_none(self) -> None:
        """ExploitDB fetch is a placeholder that returns None."""
        enricher = ExploitDBEnricher()
        result = await enricher._fetch_for_cve("CVE-2024-0001")
        assert result is None

    def test_confidence_value(self) -> None:
        assert ExploitDBEnricher.CONFIDENCE == 1.0
