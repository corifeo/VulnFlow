"""
CacheBuilder - User-facing helper for building vulnerability libraries
Tests: URL resolution, mode validation, feed_url classmethod

Layer: Helpers (user-facing)
Dependencies: None for URL tests, pipeline for integration tests
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from vulnflow.exceptions import PipelineError
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
        """build() raises PipelineError (chained to the cause) when every feed fails."""
        from vulnflow.exceptions import ExtractorError
        from vulnflow.helpers.builder import CacheBuilder

        builder = CacheBuilder(modified_only=True)
        builder._urls = ["/nonexistent/file.json"]

        with pytest.raises(PipelineError, match="All 1 feeds failed") as exc_info:
            builder.build()
        assert isinstance(exc_info.value.__cause__, ExtractorError)

    def test_build_partial_failure_returns_successful_feeds(self, nvd_sample_path):
        """One failed feed doesn't abort the build."""
        from vulnflow.helpers.builder import CacheBuilder

        builder = CacheBuilder(years=[2024], quiet=True)
        builder._urls = [str(nvd_sample_path), "/nonexistent/file.json"]

        ds = builder.build(epss=False, kev=False, cpe=False, cwe=False)

        assert len(ds) > 0

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

        monkeypatch.setattr(
            epss_mod, "DEFAULT_EPSS_URL", "tests/fixtures/epss_sample.csv"
        )

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

        monkeypatch.setattr(cwe_mod, "DEFAULT_CWE_URL", "tests/fixtures/cwe_sample.xml")

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
    @pytest.fixture(autouse=True)
    def _no_enrichment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Freshness logic needs no enrichers, and they would hit live feeds."""
        from vulnflow.helpers.builder import CacheBuilder

        monkeypatch.setattr(
            CacheBuilder, "_build_transforms", classmethod(lambda cls, **kwargs: [])
        )

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
        prov = {
            "extractors": ["test"],
            "transforms": [],
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
        }
        prov_bytes = vjson.dumps(prov).encode("utf-8")
        table = table.replace_schema_metadata({b"vulnflow:provenance": prov_bytes})
        path = tmp_path / "old.parquet"
        pq.write_table(table, path)

        builder = CacheBuilder(years=[2024])
        monkeypatch.setattr(builder, "_urls", [str(nvd_sample_path)])
        # through requires date_max; missing means rebuild
        ds = builder.ensure(path, through="2026-12-31")
        assert len(ds) == 5


class TestBuildTransforms:
    """Test _build_transforms() returns correct transform types."""

    def test_default_returns_expected_transforms(self):
        """Default args return enrichers + flatten transforms for all 5 defaults."""
        from vulnflow.helpers.builder import CacheBuilder

        transforms = CacheBuilder._build_transforms()
        names = [t.name for t in transforms]
        # Enrichers
        assert "EPSSEnricher" in names
        assert "KEVEnricher" in names
        assert "CWEEnricher" in names
        # Flatten companions
        assert "CVSSFlattenTransform" in names
        assert "EPSSFlattenTransform" in names
        assert "KEVFlattenTransform" in names
        assert "CPEFlattenTransform" in names
        assert "CWEFlattenTransform" in names
        # Should be exactly 8 transforms (3 enrichers + 5 flattens)
        assert len(transforms) == 8

    def test_all_false_returns_empty(self):
        from vulnflow.helpers.builder import CacheBuilder

        transforms = CacheBuilder._build_transforms(
            cvss=False, epss=False, kev=False, cpe=False, cwe=False
        )
        assert transforms == []

    def test_events_auto_enables_kev(self):
        """events=True should auto-enable KEV (dependency)."""
        from vulnflow.helpers.builder import CacheBuilder

        transforms = CacheBuilder._build_transforms(
            cvss=False,
            epss=False,
            kev=False,
            cpe=False,
            cwe=False,
            events=True,
        )
        names = [t.name for t in transforms]
        assert "KEVEnricher" in names
        assert "KEVFlattenTransform" in names
        assert "EventInference" in names
        # KEV must come before EventInference (dependency ordering)
        kev_idx = names.index("KEVEnricher")
        event_idx = names.index("EventInference")
        assert kev_idx < event_idx

    def test_events_with_kev_already_true_no_duplicates(self):
        """events=True + kev=True should not create duplicate KEV transforms."""
        from vulnflow.helpers.builder import CacheBuilder

        transforms = CacheBuilder._build_transforms(
            cvss=False,
            epss=False,
            kev=True,
            cpe=False,
            cwe=False,
            events=True,
        )
        names = [t.name for t in transforms]
        assert names.count("KEVEnricher") == 1
        assert names.count("KEVFlattenTransform") == 1

    def test_cvd_state_auto_enables_events_and_kev(self):
        """cvd_state=True should auto-enable events and kev."""
        from vulnflow.helpers.builder import CacheBuilder

        transforms = CacheBuilder._build_transforms(
            cvss=False,
            epss=False,
            kev=False,
            cpe=False,
            cwe=False,
            cvd_state=True,
        )
        names = [t.name for t in transforms]
        assert "KEVEnricher" in names
        assert "KEVFlattenTransform" in names
        assert "EventInference" in names
        assert "CVDStateTransform" in names
        # Ordering: KEV < EventInference < CVDStateTransform
        assert names.index("KEVEnricher") < names.index("EventInference")
        assert names.index("EventInference") < names.index("CVDStateTransform")

    def test_tags_no_auto_enable(self):
        """tags=True doesn't auto-enable anything extra (deps are all defaults)."""
        from vulnflow.helpers.builder import CacheBuilder

        transforms = CacheBuilder._build_transforms(tags=True)
        names = [t.name for t in transforms]
        assert "TagInference" in names

    def test_nvd_history(self):
        from vulnflow.helpers.builder import CacheBuilder

        transforms = CacheBuilder._build_transforms(
            cvss=False,
            epss=False,
            kev=False,
            cpe=False,
            cwe=False,
            nvd_history=True,
        )
        names = [t.name for t in transforms]
        assert len(transforms) == 1
        assert names == ["NVDHistoryEnricher"]

    def test_capec_auto_enables_cwe(self):
        """capec=True with cwe=False auto-enables CWEEnricher."""
        from vulnflow.helpers.builder import CacheBuilder

        transforms = CacheBuilder._build_transforms(
            cvss=False,
            epss=False,
            kev=False,
            cpe=False,
            cwe=False,
            capec=True,
        )
        names = [t.name for t in transforms]
        assert "CWEEnricher" in names
        assert "CAPECEnricher" in names
        assert "CAPECFlattenTransform" in names

    def test_stride_auto_enables_capec_and_cwe(self):
        """stride=True with capec=False and cwe=False enables entire chain."""
        from vulnflow.helpers.builder import CacheBuilder

        transforms = CacheBuilder._build_transforms(
            cvss=False,
            epss=False,
            kev=False,
            cpe=False,
            cwe=False,
            stride=True,
        )
        names = [t.name for t in transforms]
        assert "CWEEnricher" in names
        assert "CAPECEnricher" in names
        assert "CAPECFlattenTransform" in names
        assert "STRIDEInference" in names
        assert "STRIDEFlattenTransform" in names

    def test_stride_with_capec_no_duplicate_capec(self):
        """stride=True + capec=True should not duplicate CAPEC transforms."""
        from vulnflow.helpers.builder import CacheBuilder

        transforms = CacheBuilder._build_transforms(
            cvss=False,
            epss=False,
            kev=False,
            cpe=False,
            cwe=False,
            capec=True,
            stride=True,
        )
        names = [t.name for t in transforms]
        assert names.count("CAPECEnricher") == 1
        assert names.count("CAPECFlattenTransform") == 1


class TestQuery:
    """Test CacheBuilder.query() classmethod."""

    def test_query_creates_nvd_api_extractor(self, monkeypatch):
        """query() uses NVDAPIExtractor and returns the pipeline dataset."""
        import vulnflow.extractors.nvd_api as nvd_api_mod
        import vulnflow.pipeline as pipeline_mod
        from vulnflow.helpers.builder import CacheBuilder

        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=5)

        mock_result = MagicMock()
        mock_result.dataset = mock_dataset

        mock_pipeline_inst = MagicMock()
        mock_pipeline_inst.run.return_value = mock_result

        monkeypatch.setattr(
            nvd_api_mod, "NVDAPIExtractor", MagicMock(return_value=MagicMock())
        )
        monkeypatch.setattr(
            pipeline_mod, "VulnPipeline", MagicMock(return_value=mock_pipeline_inst)
        )

        ds = CacheBuilder.query(
            cve_id="CVE-2024-3094",
            quiet=True,
            cvss=False,
            epss=False,
            kev=False,
            cpe=False,
            cwe=False,
        )

        # Verify NVDAPIExtractor was called with correct params
        nvd_api_mod.NVDAPIExtractor.assert_called_once()
        call_kwargs = nvd_api_mod.NVDAPIExtractor.call_args[1]
        assert call_kwargs["cve_id"] == "CVE-2024-3094"
        # Verify pipeline.run() was called and result returned
        mock_pipeline_inst.run.assert_called_once()
        assert ds is mock_dataset

    def test_query_maps_params_correctly(self, monkeypatch):
        """query() maps pub_start->pub_start_date, cvss_severity->cvss_v3_severity."""
        import vulnflow.extractors.nvd_api as nvd_api_mod
        import vulnflow.pipeline as pipeline_mod
        from vulnflow.helpers.builder import CacheBuilder

        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=1)
        mock_result = MagicMock()
        mock_result.dataset = mock_dataset
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = mock_result

        monkeypatch.setattr(
            nvd_api_mod, "NVDAPIExtractor", MagicMock(return_value=MagicMock())
        )
        monkeypatch.setattr(
            pipeline_mod, "VulnPipeline", MagicMock(return_value=mock_pipeline)
        )

        CacheBuilder.query(
            pub_start="2024-01-01",
            pub_end="2024-06-30",
            cvss_severity="HIGH",
            quiet=True,
            cvss=False,
            epss=False,
            kev=False,
            cpe=False,
            cwe=False,
        )

        call_kwargs = nvd_api_mod.NVDAPIExtractor.call_args[1]
        assert call_kwargs["pub_start_date"] == "2024-01-01"
        assert call_kwargs["pub_end_date"] == "2024-06-30"
        assert call_kwargs["cvss_v3_severity"] == "HIGH"

    def test_query_raises_on_none_dataset(self, monkeypatch):
        """query() raises PipelineError when API returns no results."""
        import vulnflow.extractors.nvd_api as nvd_api_mod
        import vulnflow.pipeline as pipeline_mod
        from vulnflow.helpers.builder import CacheBuilder

        mock_result = MagicMock()
        mock_result.dataset = None
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = mock_result

        monkeypatch.setattr(
            nvd_api_mod, "NVDAPIExtractor", MagicMock(return_value=MagicMock())
        )
        monkeypatch.setattr(
            pipeline_mod, "VulnPipeline", MagicMock(return_value=mock_pipeline)
        )

        with pytest.raises(PipelineError, match="no results"):
            CacheBuilder.query(
                keyword="nonexistent",
                quiet=True,
                cvss=False,
                epss=False,
                kev=False,
                cpe=False,
                cwe=False,
            )


@pytest.mark.slow
class TestEnsureRecent:
    """Test ensure() recent=True delta merge path."""

    def test_ensure_recent_applies_delta(
        self, monkeypatch: pytest.MonkeyPatch, nvd_sample_path: Path, tmp_path: Path
    ) -> None:
        """ensure(recent=True) on fresh cache does delta merge, not full rebuild."""
        from vulnflow.helpers.builder import CacheBuilder

        builder = CacheBuilder(years=[2024])
        monkeypatch.setattr(builder, "_urls", [str(nvd_sample_path)])
        path = tmp_path / "cache.parquet"

        # First build creates cache
        ds_initial = builder.ensure(path)
        assert path.exists()
        initial_count = len(ds_initial)

        # Recent should do delta merge (create temp delta file, merge, cleanup)
        ds = builder.ensure(path, recent=True)
        assert len(ds) >= initial_count
        # Verify delta file was cleaned up
        delta_path = path.parent / f".{path.stem}_delta.parquet"
        assert not delta_path.exists()


class TestEnsureRecentOffline:
    """ensure(recent=True) with build() stubbed: no network."""

    @staticmethod
    def _cache(path: Path, nvd_sample_path: Path, modified_days_ago: float) -> None:
        import numpy as np

        ds = NVDExtractor(str(nvd_sample_path)).extract()
        stamp = np.datetime64(datetime.now() - timedelta(days=modified_days_ago), "us")
        ds.set_column("nvd_modified_at", np.full(len(ds), stamp))
        ds.to_parquet(path)

    @staticmethod
    def _stub_build(
        monkeypatch: pytest.MonkeyPatch, nvd_sample_path: Path
    ) -> list[list[str]]:
        """Record each build()'s URLs; the Modified feed returns one fixture CVE."""
        from vulnflow.helpers.builder import CacheBuilder

        calls: list[list[str]] = []

        def fake_build(self: CacheBuilder, **kwargs: object) -> object:
            calls.append(list(self._urls))
            ds = NVDExtractor(str(nvd_sample_path)).extract()
            return (
                ds.filter(ds.get_column("cve_id") == ds.get_column("cve_id")[0])
                if (self._urls == [CacheBuilder.feed_url("modified")])
                else ds
            )

        monkeypatch.setattr(CacheBuilder, "build", fake_build)
        monkeypatch.setattr(
            CacheBuilder, "_build_transforms", classmethod(lambda cls, **kw: [])
        )
        return calls

    def test_recent_on_cache_older_than_modified_window_rebuilds(
        self, monkeypatch: pytest.MonkeyPatch, nvd_sample_path: Path, tmp_path: Path
    ) -> None:
        """The Modified feed covers 8 days; an older cache gets a full build, not a delta."""
        from vulnflow.helpers.builder import CacheBuilder

        path = tmp_path / "cache.parquet"
        self._cache(path, nvd_sample_path, modified_days_ago=10)
        calls = self._stub_build(monkeypatch, nvd_sample_path)

        CacheBuilder(years=[2024]).ensure(path, recent=True)

        assert calls == [[CacheBuilder.feed_url(2024)]]

    def test_recent_on_fresh_cache_applies_delta_without_duplicates(
        self, monkeypatch: pytest.MonkeyPatch, nvd_sample_path: Path, tmp_path: Path
    ) -> None:
        from vulnflow.helpers.builder import CacheBuilder

        path = tmp_path / "cache.parquet"
        self._cache(path, nvd_sample_path, modified_days_ago=1)
        calls = self._stub_build(monkeypatch, nvd_sample_path)

        ds = CacheBuilder(years=[2024]).ensure(path, recent=True)

        assert calls == [[CacheBuilder.feed_url("modified")]]
        ids = list(ds.get_column("cve_id"))
        assert len(ids) == len(set(ids)) == 5

    def test_recent_refreshes_epss_on_every_row(
        self, monkeypatch: pytest.MonkeyPatch, nvd_sample_path: Path, tmp_path: Path
    ) -> None:
        """EPSS rescores all CVEs daily, so a delta update re-enriches the whole cache."""
        from vulnflow.dataset import VulnDataset
        from vulnflow.helpers.builder import CacheBuilder
        from vulnflow.transforms.enrichers import epss as epss_mod
        from vulnflow.transforms.enrichers.epss import EPSSEnricher
        from vulnflow.transforms.flatten import EPSSFlattenTransform

        path = tmp_path / "cache.parquet"
        self._cache(path, nvd_sample_path, modified_days_ago=1)
        cached = VulnDataset.from_parquet(path)  # cache already scored on 2024-01-15
        cached.transform(
            [
                EPSSEnricher(Path("tests/fixtures/epss_sample.csv")),
                EPSSFlattenTransform(),
            ],
            quiet=True,
        )
        cached.to_parquet(path)
        self._stub_build(monkeypatch, nvd_sample_path)
        newer = tmp_path / "epss_new.csv"
        newer.write_text(
            Path("tests/fixtures/epss_sample.csv")
            .read_text()
            .replace("2024-01-15", "2024-02-20")
        )
        monkeypatch.setattr(
            epss_mod, "DEFAULT_EPSS_URL", newer
        )  # a Path: a str default is fetched as a URL

        ds = CacheBuilder(years=[2024]).ensure(path, recent=True, kev=False)

        dates = {
            str(d)[:10]
            for d in ds.get_column("epss_date")
            if d is not None and str(d) != "NaT"
        }
        assert dates == {"2024-02-20"}


CVELIST_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "cvelist"


def _cvelist_zip(path: Path, records: list[dict[str, object]] | None = None) -> Path:
    """Write fixture records (or the given ones) into a release-style zip."""
    import json
    import zipfile

    if records is None:
        records = [
            json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(CVELIST_DIR.glob("CVE-*.json"))
        ]
    with zipfile.ZipFile(path, "w") as zf:
        for r in records:
            cve_id = r["cveMetadata"]["cveId"]  # type: ignore[index]
            zf.writestr(f"cves/{cve_id}.json", json.dumps(r))
    return path


class TestCacheBuilderCvelist:
    """CacheBuilder(source="cvelist"): release resolution, build, ensure(recent)."""

    @staticmethod
    def _releases(
        monkeypatch: pytest.MonkeyPatch, full: Path, delta: Path | None = None
    ) -> list[str]:
        """Serve fake releases whose asset URLs are local zip paths."""
        from datetime import UTC

        from vulnflow.utils import cvelist

        today = datetime.now(UTC).date().isoformat()
        assets = [
            {
                "name": f"{today}_all_CVEs_at_midnight.zip.zip",
                "browser_download_url": str(full),
            }
        ]
        if delta is not None:
            assets.append(
                {
                    "name": f"{today}_delta_CVEs_at_1700Z.zip",
                    "browser_download_url": str(delta),
                }
            )
        calls: list[str] = []

        def fake_fetch(
            timeout: float = 30.0, per_page: int = 30
        ) -> list[dict[str, object]]:
            calls.append("releases")
            return [{"tag_name": f"cve_{today}_1700Z", "assets": assets}]

        monkeypatch.setattr(cvelist, "fetch_releases", fake_fetch)
        return calls

    _FLAGS = {"epss": False, "kev": False, "cwe": False}

    def test_default_source_is_nvd(self) -> None:
        from vulnflow.helpers import CacheBuilder

        assert CacheBuilder(years=[2024])._source == "nvd"

    def test_unknown_source_raises(self) -> None:
        from vulnflow.helpers import CacheBuilder

        with pytest.raises(ValueError, match="source"):
            CacheBuilder(source="osv")

    @pytest.mark.parametrize(
        "kwargs",
        [{"years": [2024]}, {"modified_only": True}, {"include_modified": True}],
    )
    def test_cvelist_rejects_nvd_feed_options(self, kwargs: dict[str, object]) -> None:
        from vulnflow.helpers import CacheBuilder

        with pytest.raises(ValueError, match="cvelist"):
            CacheBuilder(source="cvelist", **kwargs)  # type: ignore[arg-type]

    def test_build_uses_latest_midnight_zip(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from vulnflow.helpers import CacheBuilder

        full = _cvelist_zip(tmp_path / "full.zip")
        self._releases(monkeypatch, full)

        ds = CacheBuilder(source="cvelist", quiet=True).build(**self._FLAGS)

        assert len(ds) == 8
        ids = list(ds.get_column("cve_id"))
        assert ds.get_column("cvss_source")[ids.index("CVE-2025-0168")] == "vuldb"
        assert ds.get_column("cvss_version")[ids.index("CVE-2025-0168")] == "4.0"
        assert ds.provenance["extractors"] == ["CVEListExtractor(full.zip)"]

    def _cache(self, path: Path, tmp_path: Path, day: str) -> None:
        """Cache built from a midnight zip of the given UTC day."""
        from vulnflow.extractors.cvelist import CVEListExtractor

        full = _cvelist_zip(tmp_path / f"{day}_all_CVEs_at_midnight.zip.zip")
        CVEListExtractor(full).extract().to_parquet(path)

    def test_ensure_recent_applies_todays_delta(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import json
        from datetime import UTC

        from vulnflow.helpers import CacheBuilder

        path = tmp_path / "cache.parquet"
        self._cache(path, tmp_path, datetime.now(UTC).date().isoformat())
        record = json.loads((CVELIST_DIR / "CVE-2025-0168.json").read_text("utf-8"))
        record["cveMetadata"]["dateUpdated"] = "2099-01-01T00:00:00.000Z"
        record["containers"]["cna"]["descriptions"][0]["value"] = "Updated today."
        delta = _cvelist_zip(tmp_path / "delta.zip", [record])
        self._releases(monkeypatch, tmp_path / "missing-full.zip", delta)

        ds = CacheBuilder(source="cvelist", quiet=True).ensure(
            path, recent=True, **self._FLAGS
        )

        ids = list(ds.get_column("cve_id"))
        assert len(ids) == len(set(ids)) == 8
        assert (
            ds.get_column("description")[ids.index("CVE-2025-0168")] == "Updated today."
        )

    def test_ensure_recent_on_older_snapshot_rebuilds(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Deltas start at midnight UTC: a cache from an earlier day gets a full build."""
        from vulnflow.helpers import CacheBuilder

        path = tmp_path / "cache.parquet"
        self._cache(path, tmp_path, "2026-01-01")
        full = _cvelist_zip(tmp_path / "full.zip")
        self._releases(monkeypatch, full, tmp_path / "missing-delta.zip")

        ds = CacheBuilder(source="cvelist", quiet=True).ensure(
            path, recent=True, **self._FLAGS
        )

        assert ds.provenance["extractors"] == ["CVEListExtractor(full.zip)"]
