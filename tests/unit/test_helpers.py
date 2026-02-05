"""
CacheBuilder - User-facing helper for building vulnerability libraries
Tests: URL resolution, mode validation, feed_url classmethod

Layer: Helpers (user-facing)
Dependencies: None for URL tests, pipeline for integration tests
"""
from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path

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
        assert url == "https://nvd.nist.gov/feeds/json/cve/2.0/nvdcve-2.0-Modified.json.gz"

    def test_feed_url_recent_returns_correct_url(self):
        """feed_url('recent') returns Recent feed URL."""
        from vulnflow.helpers import CacheBuilder

        url = CacheBuilder.feed_url("recent")
        assert url == "https://nvd.nist.gov/feeds/json/cve/2.0/nvdcve-2.0-Recent.json.gz"


class TestCacheBuilderBuild:
    """Test CacheBuilder.build() end-to-end with local fixture data."""

    def test_build_returns_vuln_dataset(self, nvd_sample_path):
        """build() returns a VulnDataset with records."""
        from vulnflow.helpers.builder import CacheBuilder

        # Patch _urls to use local fixture file instead of NVD URLs
        builder = CacheBuilder(years=[2024])
        builder._urls = [str(nvd_sample_path)]

        ds = builder.build(workers=1)

        assert len(ds) > 0

    def test_build_cleans_temp_dir_on_success(self, nvd_sample_path):
        """Temp directory is cleaned up after successful build."""
        import glob

        from vulnflow.helpers.builder import CacheBuilder

        builder = CacheBuilder(years=[2024])
        builder._urls = [str(nvd_sample_path)]

        builder.build(workers=1)

        # No vulnflow_cache_ dirs should remain in system temp
        temp_root = tempfile.gettempdir()
        leftover = glob.glob(str(Path(temp_root) / "vulnflow_cache_*"))
        assert len(leftover) == 0

    def test_build_cleans_temp_dir_on_failure(self):
        """Temp directory is cleaned up even when build fails."""
        import glob

        from vulnflow.exceptions import PipelineError
        from vulnflow.helpers.builder import CacheBuilder

        builder = CacheBuilder(modified_only=True)
        builder._urls = ["/nonexistent/file.json"]

        with pytest.raises(PipelineError):
            builder.build(workers=1)

        temp_root = tempfile.gettempdir()
        leftover = glob.glob(str(Path(temp_root) / "vulnflow_cache_*"))
        assert len(leftover) == 0
