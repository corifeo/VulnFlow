"""
CacheBuilder - User-facing helper for building vulnerability libraries
Tests: URL resolution, mode validation, feed_url classmethod

Layer: Helpers (user-facing)
Dependencies: None for URL tests, pipeline for integration tests
"""

from __future__ import annotations

from datetime import datetime

import pytest


class TestCacheBuilderURLResolution:
    """Test that CacheBuilder resolves correct NVD feed URLs from parameters."""

    def test_default_builds_all_years_urls(self):
        """No arguments = all years from 2002 to current year."""
        from vulnflow.helpers import CacheBuilder

        builder = CacheBuilder()
        current_year = datetime.now().year
        expected_count = current_year - 2002 + 1
        assert len(builder._urls) == expected_count
        assert builder._urls[0].endswith("nvdcve-2.0-2002.json.gz")
        assert builder._urls[-1].endswith(f"nvdcve-2.0-{current_year}.json.gz")

    def test_years_builds_specific_urls(self):
        """years=[2024, 2025] builds URLs for those years only."""
        from vulnflow.helpers import CacheBuilder

        builder = CacheBuilder(years=[2024, 2025])
        assert len(builder._urls) == 2
        assert builder._urls[0].endswith("nvdcve-2.0-2024.json.gz")
        assert builder._urls[1].endswith("nvdcve-2.0-2025.json.gz")

    def test_exclude_years_filters_from_all(self):
        """exclude_years removes specific years from full range."""
        from vulnflow.helpers import CacheBuilder

        builder = CacheBuilder(exclude_years=[2002, 2003])
        assert not any("2002" in url for url in builder._urls)
        assert not any("2003" in url for url in builder._urls)
        assert any("2004" in url for url in builder._urls)

    def test_include_modified_adds_modified_url(self):
        """include_modified=True appends Modified feed URL."""
        from vulnflow.helpers import CacheBuilder

        builder = CacheBuilder(years=[2024], include_modified=True)
        assert len(builder._urls) == 2
        assert builder._urls[0].endswith("nvdcve-2.0-2024.json.gz")
        assert builder._urls[1].endswith("nvdcve-2.0-Modified.json.gz")

    def test_modified_only_builds_single_url(self):
        """modified_only=True builds only the Modified feed URL."""
        from vulnflow.helpers import CacheBuilder

        builder = CacheBuilder(modified_only=True)
        assert len(builder._urls) == 1
        assert builder._urls[0].endswith("nvdcve-2.0-Modified.json.gz")

    def test_recent_only_builds_single_url(self):
        """recent_only=True builds only the Recent feed URL."""
        from vulnflow.helpers import CacheBuilder

        builder = CacheBuilder(recent_only=True)
        assert len(builder._urls) == 1
        assert builder._urls[0].endswith("nvdcve-2.0-Recent.json.gz")


class TestCacheBuilderValidation:
    """Test that invalid parameter combinations raise errors."""

    def test_years_and_exclude_years_mutually_exclusive(self):
        """Cannot specify both years and exclude_years."""
        from vulnflow.helpers import CacheBuilder

        with pytest.raises(ValueError, match="mutually exclusive"):
            CacheBuilder(years=[2024], exclude_years=[2002])

    def test_modified_only_and_years_mutually_exclusive(self):
        """Cannot specify modified_only with years."""
        from vulnflow.helpers import CacheBuilder

        with pytest.raises(ValueError, match="mutually exclusive"):
            CacheBuilder(modified_only=True, years=[2024])

    def test_recent_only_and_years_mutually_exclusive(self):
        """Cannot specify recent_only with years."""
        from vulnflow.helpers import CacheBuilder

        with pytest.raises(ValueError, match="mutually exclusive"):
            CacheBuilder(recent_only=True, years=[2024])

    def test_modified_only_and_recent_only_mutually_exclusive(self):
        """Cannot specify both modified_only and recent_only."""
        from vulnflow.helpers import CacheBuilder

        with pytest.raises(ValueError, match="mutually exclusive"):
            CacheBuilder(modified_only=True, recent_only=True)


class TestCacheBuilderFeedURL:
    """Test the feed_url() classmethod."""

    def test_feed_url_year_returns_correct_url(self):
        """feed_url(2024) returns year feed URL."""
        from vulnflow.helpers import CacheBuilder

        url = CacheBuilder.feed_url(2024)
        assert url == "https://nvd.nist.gov/feeds/json/cve/2.0/nvdcve-2.0-2024.json.gz"

    def test_feed_url_modified_returns_correct_url(self):
        """feed_url('modified') returns Modified feed URL."""
        from vulnflow.helpers import CacheBuilder

        url = CacheBuilder.feed_url("modified")
        assert (
            url == "https://nvd.nist.gov/feeds/json/cve/2.0/nvdcve-2.0-Modified.json.gz"
        )

    def test_feed_url_recent_returns_correct_url(self):
        """feed_url('recent') returns Recent feed URL."""
        from vulnflow.helpers import CacheBuilder

        url = CacheBuilder.feed_url("recent")
        assert (
            url == "https://nvd.nist.gov/feeds/json/cve/2.0/nvdcve-2.0-Recent.json.gz"
        )


class TestCacheBuilderBuild:
    """Test CacheBuilder.build() end-to-end with local fixture data."""

    def test_build_returns_vuln_dataset(self, nvd_sample_path):
        """build() returns a VulnDataset with records."""
        from vulnflow.helpers.builder import CacheBuilder

        # Patch _urls to use local fixture file instead of NVD URLs
        builder = CacheBuilder(years=[2024])
        builder._urls = [str(nvd_sample_path)]

        ds = builder.build(epss=False, kev=False)

        assert len(ds) > 0

    def test_build_default_flattens_cvss(self, nvd_sample_path):
        """build() defaults to cvss=True, applying CVSSFlattenTransform."""
        from vulnflow.helpers.builder import CacheBuilder

        builder = CacheBuilder(years=[2024])
        builder._urls = [str(nvd_sample_path)]

        ds = builder.build(epss=False, kev=False)

        # Default behavior: CVSSFlattenTransform adds 15 scalar columns
        assert "cvss_base_score" in ds.columns
        assert "cvss_base_severity" in ds.columns
        assert "cvss_vector_string" in ds.columns
        assert "cvss_attack_vector" in ds.columns

    def test_build_cvss_false_keeps_nested(self, nvd_sample_path):
        """build(cvss=False) keeps cvss_scores as nested JSON."""
        from vulnflow.helpers.builder import CacheBuilder

        builder = CacheBuilder(years=[2024])
        builder._urls = [str(nvd_sample_path)]

        ds = builder.build(cvss=False, epss=False, kev=False)

        # With cvss=False, only cvss_scores exists (nested JSON)
        assert "cvss_scores" in ds.columns
        assert "cvss_base_score" not in ds.columns

    def test_build_cvss_true_explicit(self, nvd_sample_path):
        """build(cvss=True) explicitly applies CVSSFlattenTransform."""
        from vulnflow.helpers.builder import CacheBuilder

        builder = CacheBuilder(years=[2024])
        builder._urls = [str(nvd_sample_path)]

        ds = builder.build(cvss=True, epss=False, kev=False)

        # CVSSFlattenTransform adds 15 scalar columns
        assert "cvss_base_score" in ds.columns
        assert "cvss_base_severity" in ds.columns

    def test_build_failure_raises(self):
        """build() raises when extraction fails."""
        from vulnflow.helpers.builder import CacheBuilder

        builder = CacheBuilder(modified_only=True)
        builder._urls = ["/nonexistent/file.json"]

        with pytest.raises((FileNotFoundError, OSError, RuntimeError)):
            builder.build()

    def test_build_rejects_workers_param(self):
        """build() does not accept workers parameter (removed in executor redesign)."""
        import inspect

        from vulnflow.helpers.builder import CacheBuilder

        sig = inspect.signature(CacheBuilder.build)
        assert "workers" not in sig.parameters

    def test_build_default_enriches_epss(self, nvd_sample_path, monkeypatch):
        """build() enriches with EPSS by default."""
        from vulnflow.helpers.builder import CacheBuilder
        from vulnflow.transforms.enrichers import epss as epss_mod

        monkeypatch.setattr(epss_mod, "DEFAULT_EPSS_URL", "tests/fixtures/epss_sample.csv")

        builder = CacheBuilder(years=[2024])
        builder._urls = [str(nvd_sample_path)]
        ds = builder.build(kev=False)

        assert "epss_probability" in ds.columns
        assert "epss_percentile" in ds.columns

    def test_build_default_enriches_kev(self, nvd_sample_path, monkeypatch):
        """build() enriches with KEV by default."""
        from vulnflow.helpers.builder import CacheBuilder
        from vulnflow.transforms.enrichers import kev as kev_mod

        monkeypatch.setattr(kev_mod, "DEFAULT_KEV_URL", "tests/fixtures/kev_sample.csv")

        builder = CacheBuilder(years=[2024])
        builder._urls = [str(nvd_sample_path)]
        ds = builder.build(epss=False)

        assert "is_kev" in ds.columns
        assert "kev_date_added" in ds.columns

    def test_build_epss_false_skips(self, nvd_sample_path):
        """build(epss=False) skips EPSS enrichment."""
        from vulnflow.helpers.builder import CacheBuilder

        builder = CacheBuilder(years=[2024])
        builder._urls = [str(nvd_sample_path)]
        ds = builder.build(epss=False, kev=False)

        assert "epss_probability" not in ds.columns

    def test_build_kev_false_skips(self, nvd_sample_path):
        """build(kev=False) skips KEV enrichment."""
        from vulnflow.helpers.builder import CacheBuilder

        builder = CacheBuilder(years=[2024])
        builder._urls = [str(nvd_sample_path)]
        ds = builder.build(epss=False, kev=False)

        assert "is_kev" not in ds.columns
