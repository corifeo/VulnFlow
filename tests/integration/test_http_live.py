"""Live HTTP integration tests - requires network access.

These tests make real HTTP requests to external services.
Run with: uv run pytest tests/integration/test_http_live.py -v

After confirmation, add @pytest.mark.skip to disable by default.
"""

from __future__ import annotations

import numpy as np
import pytest

from vulnflow.dataset import VulnDataset
from vulnflow.transforms.enrichers.epss import DEFAULT_EPSS_URL, EPSSEnricher
from vulnflow.transforms.enrichers.kev import DEFAULT_KEV_URL, KEVEnricher
from vulnflow.utils.http import HttpFetcher


def make_test_dataset(cve_ids: list[str]) -> VulnDataset:
    """Create a minimal test dataset with required columns."""
    n = len(cve_ids)
    dataset = VulnDataset()
    dataset._data = {
        "cve_id": np.array(cve_ids, dtype=object),
        "sources": np.array(["[]"] * n, dtype=object),
        "last_updated_at": np.array([None] * n, dtype=object),
    }
    return dataset


class TestLiveHttpFetcher:
    """Live tests for HttpFetcher."""

    def test_fetch_kev_csv(self, tmp_path: pytest.TempPathFactory) -> None:
        """Fetch real KEV CSV from CISA."""
        fetcher = HttpFetcher(cache_dir=tmp_path, timeout=60.0)  # type: ignore[arg-type]
        content, fmt = fetcher.fetch_sync(DEFAULT_KEV_URL, cache_ttl=None)

        assert len(content) > 1000, "KEV CSV should be substantial"
        assert fmt == "csv"
        assert b"cveID" in content, "Should contain CSV header"
        assert b"CVE-" in content, "Should contain CVE entries"

    def test_fetch_epss_csv_gz(self, tmp_path: pytest.TempPathFactory) -> None:
        """Fetch real EPSS CSV.gz from Cyentia."""
        fetcher = HttpFetcher(cache_dir=tmp_path, timeout=120.0)  # type: ignore[arg-type]
        content, fmt = fetcher.fetch_sync(DEFAULT_EPSS_URL, cache_ttl=None)

        # Content should be decompressed automatically
        assert len(content) > 10000, "EPSS CSV should be large"
        assert fmt == "csv"
        assert b"cve,epss,percentile" in content, "Should contain CSV header"
        assert b"CVE-" in content, "Should contain CVE entries"


class TestLiveKEVEnricher:
    """Live tests for KEVEnricher."""

    @pytest.fixture
    def sample_dataset(self) -> VulnDataset:
        """Create dataset with known KEV CVEs."""
        # These CVEs are in the KEV catalog
        return make_test_dataset(["CVE-2021-44228", "CVE-2023-99999", "CVE-2017-0144"])

    async def test_kev_enricher_live_fetch(self, sample_dataset: VulnDataset) -> None:
        """Enrich dataset with live KEV data."""
        enricher = KEVEnricher(cache_ttl=3600)  # Cache for 1 hour
        await enricher.transform(sample_dataset)

        # Check KEV columns were added
        assert "kev_date_added" in sample_dataset.columns
        assert "kev_vendor" in sample_dataset.columns

        # CVE-2021-44228 (Log4Shell) should be in KEV
        kev_dates = sample_dataset.get_column("kev_date_added")
        kev_vendors = sample_dataset.get_column("kev_vendor")

        # Log4Shell is definitely in KEV
        assert kev_dates[0] is not None, "CVE-2021-44228 should be in KEV"
        assert kev_vendors[0] == "Apache", "Log4Shell vendor should be Apache"

        # CVE-2023-99999 is fake, should not match
        assert kev_dates[1] is None, "Fake CVE should not match"

        # CVE-2017-0144 (EternalBlue) should be in KEV
        assert kev_dates[2] is not None, "CVE-2017-0144 should be in KEV"


class TestLiveEPSSEnricher:
    """Live tests for EPSSEnricher."""

    @pytest.fixture
    def sample_dataset(self) -> VulnDataset:
        """Create dataset with known CVEs."""
        return make_test_dataset(["CVE-2021-44228", "CVE-2023-99999", "CVE-2017-0144"])

    async def test_epss_enricher_live_fetch(self, sample_dataset: VulnDataset) -> None:
        """Enrich dataset with live EPSS data."""
        enricher = EPSSEnricher(cache_ttl=3600)  # Cache for 1 hour
        await enricher.transform(sample_dataset)

        # Check EPSS columns were added
        assert "epss_probability" in sample_dataset.columns
        assert "epss_percentile" in sample_dataset.columns
        assert "epss_date" in sample_dataset.columns

        # Get EPSS data
        probs = sample_dataset.get_column("epss_probability")
        percs = sample_dataset.get_column("epss_percentile")

        # CVE-2021-44228 (Log4Shell) should have high EPSS
        import math

        assert not math.isnan(probs[0]), "CVE-2021-44228 should have EPSS score"
        assert probs[0] > 0.5, "Log4Shell should have high exploitation probability"
        assert percs[0] > 0.9, "Log4Shell should be in top percentiles"

        # CVE-2023-99999 is fake, should have NaN
        assert math.isnan(probs[1]), "Fake CVE should not have EPSS score"

        # CVE-2017-0144 (EternalBlue) should have EPSS
        assert not math.isnan(probs[2]), "CVE-2017-0144 should have EPSS score"


class TestLiveEndToEnd:
    """End-to-end live test combining multiple enrichers."""

    async def test_full_enrichment_pipeline(self) -> None:
        """Run full enrichment with live data."""
        # Create dataset with test CVEs
        dataset = make_test_dataset(["CVE-2021-44228", "CVE-2017-0144"])

        # Run both enrichers
        kev_enricher = KEVEnricher(cache_ttl=3600)
        epss_enricher = EPSSEnricher(cache_ttl=3600)

        await kev_enricher.transform(dataset)
        await epss_enricher.transform(dataset)

        # Verify both enrichers added data
        assert "kev_date_added" in dataset.columns
        assert "epss_probability" in dataset.columns

        # Both CVEs should have KEV and EPSS data
        kev_dates = dataset.get_column("kev_date_added")
        epss_probs = dataset.get_column("epss_probability")

        import math

        for i, cve_id in enumerate(["CVE-2021-44228", "CVE-2017-0144"]):
            assert kev_dates[i] is not None, f"{cve_id} should be in KEV"
            assert not math.isnan(epss_probs[i]), f"{cve_id} should have EPSS score"
