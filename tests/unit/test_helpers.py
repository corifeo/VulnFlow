"""
CacheBuilder - User-facing helper for building vulnerability libraries
Tests: URL resolution, mode validation, feed_url classmethod

Layer: Helpers (user-facing)
Dependencies: None for URL tests, pipeline for integration tests
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from vulnflow.extractors.nvd import NVDExtractor


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

        ds = builder.build(epss=False, kev=False, cpe=False, cwe=False)

        assert len(ds) > 0

    def test_build_default_flattens_cvss(self, nvd_sample_path):
        """build() defaults to cvss=True, applying CVSSFlattenTransform."""
        from vulnflow.helpers.builder import CacheBuilder

        builder = CacheBuilder(years=[2024])
        builder._urls = [str(nvd_sample_path)]

        ds = builder.build(epss=False, kev=False, cpe=False, cwe=False)

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

        ds = builder.build(cvss=False, epss=False, kev=False, cpe=False, cwe=False)

        # With cvss=False, only cvss_scores exists (nested JSON)
        assert "cvss_scores" in ds.columns
        assert "cvss_base_score" not in ds.columns

    def test_build_cvss_true_explicit(self, nvd_sample_path):
        """build(cvss=True) explicitly applies CVSSFlattenTransform."""
        from vulnflow.helpers.builder import CacheBuilder

        builder = CacheBuilder(years=[2024])
        builder._urls = [str(nvd_sample_path)]

        ds = builder.build(cvss=True, epss=False, kev=False, cpe=False, cwe=False)

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
        ds = builder.build(kev=False, cwe=False)

        assert "epss_probability" in ds.columns
        assert "epss_percentile" in ds.columns

    def test_build_default_enriches_kev(self, nvd_sample_path, monkeypatch):
        """build() enriches with KEV by default."""
        from vulnflow.helpers.builder import CacheBuilder
        from vulnflow.transforms.enrichers import kev as kev_mod

        monkeypatch.setattr(kev_mod, "DEFAULT_KEV_URL", "tests/fixtures/kev_sample.csv")

        builder = CacheBuilder(years=[2024])
        builder._urls = [str(nvd_sample_path)]
        ds = builder.build(epss=False, cwe=False)

        assert "is_kev" in ds.columns
        assert "kev_date_added" in ds.columns

    def test_build_epss_false_skips(self, nvd_sample_path):
        """build(epss=False) skips EPSS enrichment."""
        from vulnflow.helpers.builder import CacheBuilder

        builder = CacheBuilder(years=[2024])
        builder._urls = [str(nvd_sample_path)]
        ds = builder.build(epss=False, kev=False, cpe=False, cwe=False)

        assert "epss_probability" not in ds.columns

    def test_build_kev_false_skips(self, nvd_sample_path):
        """build(kev=False) skips KEV enrichment."""
        from vulnflow.helpers.builder import CacheBuilder

        builder = CacheBuilder(years=[2024])
        builder._urls = [str(nvd_sample_path)]
        ds = builder.build(epss=False, kev=False, cpe=False, cwe=False)

        assert "is_kev" not in ds.columns

    def test_build_default_flattens_cpe(self, nvd_sample_path):
        """build() flattens CPE vendor/product columns by default."""
        from vulnflow.helpers.builder import CacheBuilder

        builder = CacheBuilder(years=[2024])
        builder._urls = [str(nvd_sample_path)]
        ds = builder.build(epss=False, kev=False, cwe=False)

        assert "cpe_vendor" in ds.columns
        assert "cpe_product" in ds.columns

    def test_build_cpe_false_skips(self, nvd_sample_path):
        """build(cpe=False) skips CPE flattening."""
        from vulnflow.helpers.builder import CacheBuilder

        builder = CacheBuilder(years=[2024])
        builder._urls = [str(nvd_sample_path)]
        ds = builder.build(epss=False, kev=False, cpe=False, cwe=False)

        assert "cpe_vendor" not in ds.columns

    def test_build_default_enriches_cwe(self, nvd_sample_path, monkeypatch):
        """build() enriches with CWE by default."""
        from vulnflow.helpers.builder import CacheBuilder
        from vulnflow.transforms.enrichers import cwe as cwe_mod

        monkeypatch.setattr(
            cwe_mod, "DEFAULT_CWE_URL", "tests/fixtures/cwe_sample.xml"
        )

        builder = CacheBuilder(years=[2024])
        builder._urls = [str(nvd_sample_path)]
        ds = builder.build(epss=False, kev=False)

        assert "cwe_id" in ds.columns
        assert "cwe_name" in ds.columns

    def test_build_cwe_false_skips(self, nvd_sample_path):
        """build(cwe=False) skips CWE enrichment."""
        from vulnflow.helpers.builder import CacheBuilder

        builder = CacheBuilder(years=[2024])
        builder._urls = [str(nvd_sample_path)]
        ds = builder.build(epss=False, kev=False, cpe=False, cwe=False)

        assert "cwe_id" not in ds.columns


class TestParseDuration:
    @pytest.mark.parametrize(
        "input_val,expected_days",
        [
            ("7d", 7),
            ("30d", 30),
            ("1d", 1),
            ("24h", 1),
            ("48h", 2),
            ("12h", 0.5),
        ],
    )
    def test_parse_duration_strings(self, input_val: str, expected_days: float) -> None:
        from vulnflow.helpers.builder import _parse_duration

        result = _parse_duration(input_val)
        assert result == timedelta(days=expected_days)

    def test_parse_duration_timedelta_passthrough(self) -> None:
        from vulnflow.helpers.builder import _parse_duration

        td = timedelta(days=3)
        assert _parse_duration(td) is td

    def test_parse_duration_invalid_raises(self) -> None:
        from vulnflow.helpers.builder import _parse_duration

        with pytest.raises(ValueError, match="Cannot parse duration"):
            _parse_duration("invalid")


class TestReadParquetProvenance:
    def test_reads_provenance_without_loading_data(
        self, tmp_path: Path, nvd_sample_path: Path
    ) -> None:
        from vulnflow.helpers.builder import _read_parquet_provenance

        ds = NVDExtractor(str(nvd_sample_path)).extract()
        ds.to_parquet(tmp_path / "test.parquet")
        prov = _read_parquet_provenance(tmp_path / "test.parquet")
        assert prov is not None
        assert "extractors" in prov
        assert "created_at" in prov

    def test_returns_none_for_missing_file(self, tmp_path: Path) -> None:
        from vulnflow.helpers.builder import _read_parquet_provenance

        assert _read_parquet_provenance(tmp_path / "missing.parquet") is None


class TestCacheBuilderEnsure:
    def test_ensure_builds_when_file_missing(
        self, monkeypatch: pytest.MonkeyPatch, nvd_sample_path: Path, tmp_path: Path
    ) -> None:
        """ensure() does a full build when file doesn't exist."""
        from vulnflow.helpers.builder import CacheBuilder

        builder = CacheBuilder(years=[2024])
        monkeypatch.setattr(builder, "_urls", [str(nvd_sample_path)])
        path = tmp_path / "cache.parquet"
        ds = builder.ensure(path, max_age="7d")
        assert len(ds) == 5
        assert path.exists()

    def test_ensure_returns_cached_when_fresh(
        self, monkeypatch: pytest.MonkeyPatch, nvd_sample_path: Path, tmp_path: Path
    ) -> None:
        """ensure() loads from cache when file is fresh."""
        from vulnflow.helpers.builder import CacheBuilder

        builder = CacheBuilder(years=[2024])
        monkeypatch.setattr(builder, "_urls", [str(nvd_sample_path)])
        path = tmp_path / "cache.parquet"
        ds1 = builder.ensure(path, max_age="7d")
        ds2 = builder.ensure(path, max_age="7d")
        assert len(ds2) == len(ds1)

    def test_ensure_rebuilds_when_stale(
        self, monkeypatch: pytest.MonkeyPatch, nvd_sample_path: Path, tmp_path: Path
    ) -> None:
        """ensure() rebuilds when max_age exceeded."""
        from vulnflow.helpers.builder import CacheBuilder

        builder = CacheBuilder(years=[2024])
        monkeypatch.setattr(builder, "_urls", [str(nvd_sample_path)])
        path = tmp_path / "cache.parquet"
        builder.ensure(path, max_age="7d")
        # 0h = always stale
        ds = builder.ensure(path, max_age="0h")
        assert len(ds) == 5

    def test_ensure_accepts_timedelta(
        self, monkeypatch: pytest.MonkeyPatch, nvd_sample_path: Path, tmp_path: Path
    ) -> None:
        """ensure() accepts timedelta for max_age."""
        from vulnflow.helpers.builder import CacheBuilder

        builder = CacheBuilder(years=[2024])
        monkeypatch.setattr(builder, "_urls", [str(nvd_sample_path)])
        path = tmp_path / "cache.parquet"
        ds = builder.ensure(path, max_age=timedelta(days=7))
        assert len(ds) == 5

    def test_ensure_through_triggers_rebuild(
        self, monkeypatch: pytest.MonkeyPatch, nvd_sample_path: Path, tmp_path: Path
    ) -> None:
        """ensure(through=future_date) triggers rebuild when date_max is too old."""
        from vulnflow.helpers.builder import CacheBuilder

        builder = CacheBuilder(years=[2024])
        monkeypatch.setattr(builder, "_urls", [str(nvd_sample_path)])
        path = tmp_path / "cache.parquet"
        builder.ensure(path)
        # Sample data has dates in 2024; asking for 2026 should trigger rebuild
        ds = builder.ensure(path, through="2026-12-31")
        assert len(ds) > 0

    def test_ensure_with_no_args_returns_cached(
        self, monkeypatch: pytest.MonkeyPatch, nvd_sample_path: Path, tmp_path: Path
    ) -> None:
        """ensure() with no freshness args always returns cached data."""
        from vulnflow.helpers.builder import CacheBuilder

        builder = CacheBuilder(years=[2024])
        monkeypatch.setattr(builder, "_urls", [str(nvd_sample_path)])
        path = tmp_path / "cache.parquet"
        builder.ensure(path)
        ds = builder.ensure(path)
        assert len(ds) == 5

    def test_ensure_path_as_string(
        self, monkeypatch: pytest.MonkeyPatch, nvd_sample_path: Path, tmp_path: Path
    ) -> None:
        """ensure() accepts string path."""
        from vulnflow.helpers.builder import CacheBuilder

        builder = CacheBuilder(years=[2024])
        monkeypatch.setattr(builder, "_urls", [str(nvd_sample_path)])
        path = str(tmp_path / "cache.parquet")
        ds = builder.ensure(path, max_age="7d")
        assert len(ds) == 5

    def test_ensure_old_parquet_without_date_range(
        self, monkeypatch: pytest.MonkeyPatch, nvd_sample_path: Path, tmp_path: Path
    ) -> None:
        """ensure(through=...) falls back to rebuild when date_max missing."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        from vulnflow.helpers.builder import CacheBuilder
        from vulnflow.utils import json as vjson

        # Write a parquet file without date_min/date_max in provenance
        table = pa.table({"cve_id": ["CVE-2024-0001"]})
        prov = {"extractors": ["test"], "transforms": [], "created_at": "2026-01-01T00:00:00+00:00", "updated_at": "2026-01-01T00:00:00+00:00"}
        prov_bytes = vjson.dumps(prov).encode("utf-8")
        table = table.replace_schema_metadata({b"vulnflow:provenance": prov_bytes})
        path = tmp_path / "old.parquet"
        pq.write_table(table, path)

        builder = CacheBuilder(years=[2024])
        monkeypatch.setattr(builder, "_urls", [str(nvd_sample_path)])
        # through requires date_max; missing means rebuild
        ds = builder.ensure(path, through="2026-12-31")
        assert len(ds) == 5
