"""
HTTP Utilities - Tests for is_url, detect_format, HttpFetcher, and enricher/extractor HTTP integration
Tests: URL detection, format detection, caching, decompression, retry, mocked HTTP calls

Layer: Utils/HTTP
Dependencies: httpx, utils.http
"""

from __future__ import annotations

import gzip
import time
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from vulnflow.extractors.nvd import NVDExtractor
from vulnflow.transforms.enrichers.epss import DEFAULT_EPSS_URL, EPSSEnricher
from vulnflow.transforms.enrichers.kev import DEFAULT_KEV_URL, KEVEnricher
from vulnflow.utils.http import HttpError, HttpFetcher, detect_format, is_url

# =============================================================================
# is_url() Tests
# =============================================================================


class TestIsUrl:
    """Tests for is_url() function."""

    def test_http_url(self) -> None:
        assert is_url("http://example.com/file.json") is True

    def test_https_url(self) -> None:
        assert is_url("https://example.com/file.json") is True

    def test_file_path_unix(self) -> None:
        assert is_url("/path/to/file.json") is False

    def test_file_path_windows(self) -> None:
        assert is_url("C:\\path\\to\\file.json") is False

    def test_relative_path(self) -> None:
        assert is_url("./data/file.json") is False

    def test_empty_string(self) -> None:
        assert is_url("") is False


# =============================================================================
# detect_format() Tests
# =============================================================================


class TestDetectFormat:
    """Tests for detect_format() function."""

    def test_json_extension(self) -> None:
        assert detect_format("https://example.com/data.json") == "json"

    def test_jsonl_extension(self) -> None:
        assert detect_format("https://example.com/data.jsonl") == "jsonl"

    def test_ndjson_extension(self) -> None:
        assert detect_format("https://example.com/data.ndjson") == "jsonl"

    def test_csv_extension(self) -> None:
        assert detect_format("https://example.com/data.csv") == "csv"

    def test_json_gz_extension(self) -> None:
        # .gz should be stripped before checking extension
        assert detect_format("https://example.com/data.json.gz") == "json"

    def test_csv_gz_extension(self) -> None:
        assert detect_format("https://example.com/data.csv.gz") == "csv"

    def test_content_type_json(self) -> None:
        assert detect_format("https://example.com/data", "application/json") == "json"

    def test_content_type_ndjson(self) -> None:
        assert (
            detect_format("https://example.com/data", "application/x-ndjson") == "jsonl"
        )

    def test_content_type_csv(self) -> None:
        assert detect_format("https://example.com/data", "text/csv") == "csv"

    def test_content_sniffing_json_object(self) -> None:
        content = b'{"key": "value"}'
        assert detect_format("https://example.com/data", None, content) == "json"

    def test_content_sniffing_json_array(self) -> None:
        content = b'[{"key": "value"}]'
        assert detect_format("https://example.com/data", None, content) == "json"

    def test_content_sniffing_csv(self) -> None:
        content = b"cve,score\nCVE-2021-1234,9.8"
        assert detect_format("https://example.com/data", None, content) == "csv"

    def test_content_sniffing_csv_quoted(self) -> None:
        content = b'"cve","score"\n"CVE-2021-1234","9.8"'
        assert detect_format("https://example.com/data", None, content) == "csv"

    def test_content_sniffing_with_bom(self) -> None:
        # UTF-8 BOM + JSON
        content = b'\xef\xbb\xbf{"key": "value"}'
        assert detect_format("https://example.com/data", None, content) == "json"

    def test_content_sniffing_with_whitespace(self) -> None:
        content = b'  \n\t{"key": "value"}'
        assert detect_format("https://example.com/data", None, content) == "json"

    def test_unknown_format(self) -> None:
        assert detect_format("https://example.com/data") == "unknown"

    def test_extension_takes_precedence(self) -> None:
        # Even if content looks like CSV, extension wins
        csv_content = b"cve,score\nCVE-2021-1234,9.8"
        assert (
            detect_format("https://example.com/data.json", None, csv_content) == "json"
        )


# =============================================================================
# HttpFetcher Unit Tests
# =============================================================================


class TestHttpFetcher:
    """Tests for HttpFetcher class."""

    def test_init_default_cache_dir(self) -> None:
        fetcher = HttpFetcher()
        assert fetcher.cache_dir == Path(".cache")
        assert fetcher.timeout == 120.0

    def test_init_custom_cache_dir(self, tmp_path: Path) -> None:
        fetcher = HttpFetcher(cache_dir=tmp_path / "my_cache", timeout=30.0)
        assert fetcher.cache_dir == tmp_path / "my_cache"
        assert fetcher.timeout == 30.0

    def test_cache_path_generates_hash(self, tmp_path: Path) -> None:
        fetcher = HttpFetcher(cache_dir=tmp_path)
        path1 = fetcher._cache_path("https://example.com/data.json")
        path2 = fetcher._cache_path("https://example.com/other.json")
        # Different URLs should produce different cache paths
        assert path1 != path2
        # Same URL should produce same cache path
        assert path1 == fetcher._cache_path("https://example.com/data.json")

    def test_write_and_read_cache(self, tmp_path: Path) -> None:
        fetcher = HttpFetcher(cache_dir=tmp_path)
        url = "https://example.com/data.json"
        content = b"test content"

        # Write to cache
        fetcher._write_cache(url, content)

        # Read from cache (within TTL)
        result = fetcher._read_cache(url, ttl=3600)
        assert result == content

    def test_read_cache_expired(self, tmp_path: Path) -> None:
        fetcher = HttpFetcher(cache_dir=tmp_path)
        url = "https://example.com/data.json"
        content = b"test content"

        # Write to cache
        fetcher._write_cache(url, content)

        # Modify mtime to be old
        cache_path = fetcher._cache_path(url)
        old_time = time.time() - 7200  # 2 hours ago
        import os

        os.utime(cache_path, (old_time, old_time))

        # Read should return None (expired)
        result = fetcher._read_cache(url, ttl=3600)
        assert result is None

    def test_read_cache_missing(self, tmp_path: Path) -> None:
        fetcher = HttpFetcher(cache_dir=tmp_path)
        result = fetcher._read_cache("https://nonexistent.com/data.json", ttl=3600)
        assert result is None

    def test_maybe_decompress_gzip_by_magic(self, tmp_path: Path) -> None:
        fetcher = HttpFetcher(cache_dir=tmp_path)
        original = b"test content for gzip"
        compressed = gzip.compress(original)

        result = fetcher._maybe_decompress(compressed, "https://example.com/data")
        assert result == original

    def test_maybe_decompress_gzip_by_extension(self, tmp_path: Path) -> None:
        fetcher = HttpFetcher(cache_dir=tmp_path)
        original = b"test content for gzip"
        compressed = gzip.compress(original)

        result = fetcher._maybe_decompress(
            compressed, "https://example.com/data.csv.gz"
        )
        assert result == original

    def test_maybe_decompress_not_gzip(self, tmp_path: Path) -> None:
        fetcher = HttpFetcher(cache_dir=tmp_path)
        content = b"plain text content"

        result = fetcher._maybe_decompress(content, "https://example.com/data.txt")
        assert result == content

    def test_cache_creates_directory(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "nested" / "cache" / "dir"
        fetcher = HttpFetcher(cache_dir=cache_dir)

        fetcher._write_cache("https://example.com/data.json", b"content")
        assert cache_dir.exists()

    async def test_fetch_decompress_false_skips_decompression(
        self, tmp_path: Path
    ) -> None:
        original = b"test content for gzip"
        compressed = gzip.compress(original)
        ok_response = _make_response(200, compressed)

        fetcher = HttpFetcher(cache_dir=tmp_path, retries=1)

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = ok_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=mock_client
            )
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            content, fmt = await fetcher.fetch(
                "https://example.com/data.json.gz", decompress=False
            )

        # Should return compressed bytes, not decompressed
        assert content == compressed
        assert content != original


# =============================================================================
# KEVEnricher HTTP Tests
# =============================================================================


class TestKEVEnricherHTTP:
    """Tests for KEVEnricher HTTP functionality."""

    @pytest.fixture
    def kev_csv_content(self) -> str:
        return """cveID,vendorProject,product,dateAdded,dueDate,requiredAction,knownRansomwareCampaignUse
CVE-2021-44228,Apache,Log4j,2021-12-10,2021-12-24,Apply updates,Known
CVE-2023-12345,Microsoft,Exchange,2023-06-15,2023-07-01,Apply patches,Unknown"""

    @pytest.fixture
    def kev_json_content(self) -> bytes:
        import json

        data = {
            "catalogVersion": "2024.01.01",
            "vulnerabilities": [
                {
                    "cveID": "CVE-2021-44228",
                    "vendorProject": "Apache",
                    "product": "Log4j",
                    "dateAdded": "2021-12-10",
                    "dueDate": "2021-12-24",
                    "requiredAction": "Apply updates",
                    "knownRansomwareCampaignUse": "Known",
                },
            ],
        }
        return json.dumps(data).encode()

    def test_kev_enricher_default_url(self) -> None:
        enricher = KEVEnricher()
        assert enricher.source == DEFAULT_KEV_URL

    def test_kev_enricher_custom_url(self) -> None:
        url = "https://custom.example.com/kev.csv"
        enricher = KEVEnricher(source=url)
        assert enricher.source == url

    def test_kev_enricher_file_path(self, tmp_path: Path) -> None:
        path = tmp_path / "kev.csv"
        path.touch()
        enricher = KEVEnricher(source=str(path))
        assert enricher.source == path

    async def test_kev_parse_csv(self, kev_csv_content: str) -> None:
        enricher = KEVEnricher(source="https://example.com/kev.csv")
        data = enricher._parse_kev_csv(kev_csv_content)

        assert "CVE-2021-44228" in data
        assert data["CVE-2021-44228"]["vendor"] == "Apache"
        assert data["CVE-2021-44228"]["product"] == "Log4j"
        assert data["CVE-2021-44228"]["date_added"] == datetime(
            2021, 12, 10, tzinfo=UTC
        )

    async def test_kev_parse_json(self, kev_json_content: bytes) -> None:
        enricher = KEVEnricher(source="https://example.com/kev.json")
        data = enricher._parse_kev_json(kev_json_content)

        assert "CVE-2021-44228" in data
        assert data["CVE-2021-44228"]["vendor"] == "Apache"

    async def test_kev_load_from_url_mocked(self, kev_csv_content: str) -> None:
        enricher = KEVEnricher()

        # Mock the HttpFetcher
        mock_fetcher = AsyncMock()
        mock_fetcher.fetch.return_value = (kev_csv_content.encode(), "csv")
        enricher._http_fetcher = mock_fetcher

        data = await enricher._load_kev_from_url(DEFAULT_KEV_URL)

        assert "CVE-2021-44228" in data
        mock_fetcher.fetch.assert_called_once()


# =============================================================================
# EPSSEnricher HTTP Tests
# =============================================================================


class TestEPSSEnricherHTTP:
    """Tests for EPSSEnricher HTTP functionality."""

    @pytest.fixture
    def epss_csv_content(self) -> str:
        return """#model_version:v2023.03.01,score_date:2024-01-15
cve,epss,percentile
CVE-2021-44228,0.97565,0.99970
CVE-2023-12345,0.00234,0.45000"""

    def test_epss_enricher_default_url(self) -> None:
        enricher = EPSSEnricher()
        assert enricher.source == DEFAULT_EPSS_URL

    def test_epss_enricher_custom_url(self) -> None:
        url = "https://custom.example.com/epss.csv.gz"
        enricher = EPSSEnricher(source=url)
        assert enricher.source == url

    async def test_epss_parse_csv(self, epss_csv_content: str) -> None:
        enricher = EPSSEnricher(source="https://example.com/epss.csv")
        data = enricher._parse_epss_csv(epss_csv_content)

        assert "CVE-2021-44228" in data
        assert data["CVE-2021-44228"]["probability"] == pytest.approx(0.97565)
        assert data["CVE-2021-44228"]["percentile"] == pytest.approx(0.9997)
        assert data["CVE-2021-44228"]["date"] == datetime(2024, 1, 15, tzinfo=UTC)

    async def test_epss_load_from_url_mocked(self, epss_csv_content: str) -> None:
        enricher = EPSSEnricher()

        # Mock the HttpFetcher
        mock_fetcher = AsyncMock()
        mock_fetcher.fetch.return_value = (epss_csv_content.encode(), "csv")
        enricher._http_fetcher = mock_fetcher

        data = await enricher._load_epss_from_url(DEFAULT_EPSS_URL)

        assert "CVE-2021-44228" in data
        mock_fetcher.fetch.assert_called_once()


# =============================================================================
# NVDExtractor HTTP Tests
# =============================================================================


class TestNVDExtractorHTTP:
    """Tests for NVDExtractor HTTP functionality."""

    @pytest.fixture
    def nvd_json_content(self) -> bytes:
        import json

        data = {
            "vulnerabilities": [
                {
                    "cve": {
                        "id": "CVE-2021-44228",
                        "descriptions": [{"lang": "en", "value": "Log4Shell"}],
                        "published": "2021-12-10T10:00:00.000",
                        "lastModified": "2024-01-01T12:00:00.000",
                        "metrics": {},
                        "weaknesses": [],
                        "configurations": [],
                        "references": [],
                    }
                }
            ]
        }
        return json.dumps(data).encode()

    @pytest.fixture
    def nvd_jsonl_content(self) -> bytes:
        import json

        lines = [
            json.dumps(
                {
                    "cve": {
                        "id": "CVE-2021-44228",
                        "descriptions": [{"lang": "en", "value": "Log4Shell"}],
                        "published": "2021-12-10T10:00:00.000",
                        "lastModified": "2024-01-01T12:00:00.000",
                        "metrics": {},
                        "weaknesses": [],
                        "configurations": [],
                        "references": [],
                    }
                }
            ),
            json.dumps(
                {
                    "cve": {
                        "id": "CVE-2023-12345",
                        "descriptions": [{"lang": "en", "value": "Test CVE"}],
                        "published": "2023-06-15T10:00:00.000",
                        "lastModified": "2023-06-20T12:00:00.000",
                        "metrics": {},
                        "weaknesses": [],
                        "configurations": [],
                        "references": [],
                    }
                }
            ),
        ]
        return "\n".join(lines).encode()

    def test_nvd_parse_json(self, nvd_json_content: bytes) -> None:
        extractor = NVDExtractor("https://example.com/nvd.json")
        vulns = extractor._parse_json(nvd_json_content)

        assert len(vulns) == 1
        assert vulns[0]["cve"]["id"] == "CVE-2021-44228"

    def test_nvd_parse_jsonl(self, nvd_jsonl_content: bytes) -> None:
        extractor = NVDExtractor("https://example.com/nvd.jsonl")
        vulns = extractor._parse_jsonl(nvd_jsonl_content)

        assert len(vulns) == 2
        assert vulns[0]["cve"]["id"] == "CVE-2021-44228"
        assert vulns[1]["cve"]["id"] == "CVE-2023-12345"

    def test_nvd_fetch_raw_from_url_mocked(self, nvd_json_content: bytes) -> None:
        with patch.object(HttpFetcher, "fetch_sync") as mock_fetch:
            mock_fetch.return_value = (nvd_json_content, "json")

            extractor = NVDExtractor("https://example.com/nvd.json")
            content = extractor._fetch_raw()

            assert content == nvd_json_content
            # Verify decompress=False is passed
            mock_fetch.assert_called_once()
            _, kwargs = mock_fetch.call_args
            assert kwargs.get("decompress") is False

    def test_nvd_extract_from_url_mocked(self, nvd_json_content: bytes) -> None:
        with patch.object(HttpFetcher, "fetch_sync") as mock_fetch:
            mock_fetch.return_value = (nvd_json_content, "json")

            extractor = NVDExtractor("https://example.com/nvd.json")
            dataset = extractor.extract()

            assert len(dataset) == 1
            cve_ids = dataset.get_column("cve_id")
            assert cve_ids[0] == "CVE-2021-44228"

    def test_nvd_extract_populates_timings(self, nvd_json_content: bytes) -> None:
        with patch.object(HttpFetcher, "fetch_sync") as mock_fetch:
            mock_fetch.return_value = (nvd_json_content, "json")

            extractor = NVDExtractor("https://example.com/nvd.json")
            extractor.extract()

            assert hasattr(extractor, "_timings")
            timings = extractor._timings
            assert "download" in timings
            assert "decompress" in timings
            assert "json" in timings
            assert "populate" in timings
            assert all(isinstance(v, float) for v in timings.values())
            assert all(v >= 0 for v in timings.values())

    def test_nvd_load_gzip_file(self, tmp_path: Path, nvd_json_content: bytes) -> None:
        # Create gzipped file
        gz_path = tmp_path / "nvd.json.gz"
        gz_path.write_bytes(gzip.compress(nvd_json_content))

        extractor = NVDExtractor(str(gz_path))
        raw = extractor._fetch_raw()
        content = extractor._decompress(raw)

        assert content == nvd_json_content

    def test_nvd_jsonl_path_detection(self) -> None:
        extractor = NVDExtractor("https://example.com/data.jsonl")
        # _parse_content should detect JSONL from extension
        assert extractor.path.lower().endswith(".jsonl")

    def test_nvd_jsonl_gz_path_detection(self) -> None:
        extractor = NVDExtractor("https://example.com/data.jsonl.gz")
        # Should strip .gz and detect .jsonl
        path = extractor.path.lower()
        assert path.endswith(".jsonl.gz")


# =============================================================================
# HttpFetcher Integration Tests
# =============================================================================


class TestHttpFetcherIntegration:
    """Tests for HttpFetcher with real file operations."""

    def test_gzip_decompression_integration(self, tmp_path: Path) -> None:
        original = b'{"test": "data"}'
        compressed = gzip.compress(original)

        # Write compressed data
        gz_file = tmp_path / "test.json.gz"
        gz_file.write_bytes(compressed)

        # Read and decompress
        fetcher = HttpFetcher(cache_dir=tmp_path / "cache")
        result = fetcher._maybe_decompress(
            compressed, "https://example.com/test.json.gz"
        )

        assert result == original

    def test_cache_roundtrip_integration(self, tmp_path: Path) -> None:
        url = "https://example.com/data.json"
        content = b'{"cached": true}'

        fetcher = HttpFetcher(cache_dir=tmp_path / "cache")

        # Write to cache
        fetcher._write_cache(url, content)

        # Read from cache
        result = fetcher._read_cache(url, ttl=3600)

        assert result == content

        # Verify cache file exists
        cache_path = fetcher._cache_path(url)
        assert cache_path.exists()


# =============================================================================
# HttpFetcher Retry Tests
# =============================================================================


def _make_response(
    status_code: int = 200,
    content: bytes = b'{"ok": true}',
    content_type: str = "application/json",
) -> httpx.Response:
    """Build a fake httpx.Response for testing."""
    response = httpx.Response(
        status_code=status_code,
        content=content,
        headers={"content-type": content_type},
        request=httpx.Request("GET", "https://example.com/data.json"),
    )
    return response


class TestHttpFetcherRetry:
    """Tests for HttpFetcher retry logic."""

    def test_init_stores_retry_params(self, tmp_path: Path) -> None:
        fetcher = HttpFetcher(cache_dir=tmp_path, retries=5, backoff_base=2.0)
        assert fetcher._retries == 5
        assert fetcher._backoff_base == 2.0

    def test_init_default_retry_params(self) -> None:
        fetcher = HttpFetcher()
        assert fetcher._retries == 3
        assert fetcher._backoff_base == 1.0

    async def test_fetch_succeeds_on_first_attempt(self, tmp_path: Path) -> None:
        fetcher = HttpFetcher(cache_dir=tmp_path, retries=3, backoff_base=0.0)
        ok_response = _make_response(200, b'{"data": 1}')

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = ok_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=mock_client
            )
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            content, fmt = await fetcher.fetch("https://example.com/data.json")

        assert content == b'{"data": 1}'
        assert fmt == "json"
        assert mock_client.get.call_count == 1

    async def test_fetch_retries_on_connect_error(self, tmp_path: Path) -> None:
        fetcher = HttpFetcher(cache_dir=tmp_path, retries=3, backoff_base=0.0)
        ok_response = _make_response(200, b'{"ok": true}')

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.side_effect = [
                httpx.ConnectError("Connection refused"),
                ok_response,
            ]
            mock_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=mock_client
            )
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            content, fmt = await fetcher.fetch("https://example.com/data.json")

        assert content == b'{"ok": true}'
        assert mock_client.get.call_count == 2

    async def test_fetch_retries_on_timeout(self, tmp_path: Path) -> None:
        fetcher = HttpFetcher(cache_dir=tmp_path, retries=3, backoff_base=0.0)
        ok_response = _make_response(200, b'{"ok": true}')

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.side_effect = [
                httpx.ReadTimeout("Read timed out"),
                ok_response,
            ]
            mock_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=mock_client
            )
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            content, fmt = await fetcher.fetch("https://example.com/data.json")

        assert content == b'{"ok": true}'
        assert mock_client.get.call_count == 2

    async def test_fetch_retries_on_5xx_error(self, tmp_path: Path) -> None:
        fetcher = HttpFetcher(cache_dir=tmp_path, retries=3, backoff_base=0.0)
        error_response = _make_response(503, b"Service Unavailable")
        ok_response = _make_response(200, b'{"ok": true}')

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            # First call returns 503, second call succeeds
            mock_client.get.side_effect = [error_response, ok_response]
            mock_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=mock_client
            )
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            content, fmt = await fetcher.fetch("https://example.com/data.json")

        assert content == b'{"ok": true}'
        assert mock_client.get.call_count == 2

    async def test_fetch_raises_immediately_on_4xx(self, tmp_path: Path) -> None:
        fetcher = HttpFetcher(cache_dir=tmp_path, retries=3, backoff_base=0.0)
        error_response = _make_response(404, b"Not Found")

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = error_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=mock_client
            )
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            with pytest.raises(HttpError, match="Failed to fetch"):
                await fetcher.fetch("https://example.com/data.json")

        # Should not retry on 4xx
        assert mock_client.get.call_count == 1

    async def test_fetch_exhausts_retries_raises(self, tmp_path: Path) -> None:
        fetcher = HttpFetcher(cache_dir=tmp_path, retries=3, backoff_base=0.0)

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.side_effect = httpx.ConnectError("Connection refused")
            mock_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=mock_client
            )
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            with pytest.raises(HttpError, match="Failed to fetch"):
                await fetcher.fetch("https://example.com/data.json")

        assert mock_client.get.call_count == 3

    async def test_fetch_retries_1_means_no_retry(self, tmp_path: Path) -> None:
        fetcher = HttpFetcher(cache_dir=tmp_path, retries=1, backoff_base=0.0)

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.side_effect = httpx.ConnectError("Connection refused")
            mock_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=mock_client
            )
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            with pytest.raises(HttpError):
                await fetcher.fetch("https://example.com/data.json")

        # retries=1 means one attempt total, no retries
        assert mock_client.get.call_count == 1

    async def test_fetch_5xx_on_last_attempt_raises(self, tmp_path: Path) -> None:
        fetcher = HttpFetcher(cache_dir=tmp_path, retries=2, backoff_base=0.0)
        error_response = _make_response(502, b"Bad Gateway")

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = error_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=mock_client
            )
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            with pytest.raises(HttpError, match="Failed to fetch"):
                await fetcher.fetch("https://example.com/data.json")

        assert mock_client.get.call_count == 2

    async def test_fetch_backoff_timing(self, tmp_path: Path) -> None:
        fetcher = HttpFetcher(cache_dir=tmp_path, retries=3, backoff_base=0.01)
        ok_response = _make_response(200, b'{"ok": true}')

        with (
            patch("httpx.AsyncClient") as mock_client_cls,
            patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            mock_client = AsyncMock()
            mock_client.get.side_effect = [
                httpx.ConnectError("fail 1"),
                httpx.ConnectError("fail 2"),
                ok_response,
            ]
            mock_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=mock_client
            )
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            content, fmt = await fetcher.fetch("https://example.com/data.json")

        assert content == b'{"ok": true}'
        # Two failures before success: backoff_base * 2^0, backoff_base * 2^1
        assert mock_sleep.call_count == 2
        assert mock_sleep.call_args_list[0].args[0] == pytest.approx(0.01)
        assert mock_sleep.call_args_list[1].args[0] == pytest.approx(0.02)

    async def test_fetch_logs_warning_on_retry(self, tmp_path: Path) -> None:
        fetcher = HttpFetcher(cache_dir=tmp_path, retries=2, backoff_base=0.0)
        ok_response = _make_response(200, b'{"ok": true}')

        with (
            patch("httpx.AsyncClient") as mock_client_cls,
            patch("vulnflow.utils.http.logger") as mock_logger,
        ):
            mock_client = AsyncMock()
            mock_client.get.side_effect = [
                httpx.ConnectError("Connection refused"),
                ok_response,
            ]
            mock_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=mock_client
            )
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            await fetcher.fetch("https://example.com/data.json")

        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args.args
        # logger.warning uses %-formatting: format_str, attempt+1, retries, url, exc, wait
        assert "HTTP fetch attempt" in call_args[0]
        assert call_args[1] == 1  # attempt + 1
        assert call_args[2] == 2  # retries


# =============================================================================
# HttpFetcher Streaming Progress Tests
# =============================================================================


class TestHttpFetcherStreamingProgress:
    """Tests for HttpFetcher streaming download with progress bar."""

    async def test_fetch_with_progress_uses_streaming(self, tmp_path: Path) -> None:
        """When progress_desc is non-empty, fetch() uses client.stream() instead of .get()."""
        fetcher = HttpFetcher(cache_dir=tmp_path, retries=1)
        body = b'{"data": 1}'

        # Build a mock streaming response
        mock_stream_response = MagicMock()
        mock_stream_response.raise_for_status = MagicMock()
        mock_stream_response.headers = {
            "content-length": str(len(body)),
            "content-type": "application/json",
        }
        mock_stream_response.aiter_bytes = lambda chunk_size: _async_iter([body])

        # stream() must return a sync context manager (not a coroutine),
        # whose __aenter__/__aexit__ are async.
        stream_ctx = MagicMock()
        stream_ctx.__aenter__ = AsyncMock(return_value=mock_stream_response)
        stream_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.stream = MagicMock(return_value=stream_ctx)
            mock_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=mock_client
            )
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            content, fmt = await fetcher.fetch(
                "https://example.com/data.json", progress_desc="Downloading test"
            )

        assert content == body
        assert fmt == "json"
        mock_client.stream.assert_called_once_with(
            "GET", "https://example.com/data.json"
        )
        mock_client.get.assert_not_called()

    async def test_fetch_without_progress_uses_get(self, tmp_path: Path) -> None:
        """Default progress_desc='' uses client.get() (regression guard)."""
        fetcher = HttpFetcher(cache_dir=tmp_path, retries=1)
        ok_response = _make_response(200, b'{"data": 1}')

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = ok_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=mock_client
            )
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            content, fmt = await fetcher.fetch("https://example.com/data.json")

        assert content == b'{"data": 1}'
        mock_client.get.assert_called_once()

    async def test_fetch_sync_passes_progress_desc(self, tmp_path: Path) -> None:
        """fetch_sync() forwards progress_desc to fetch()."""
        fetcher = HttpFetcher(cache_dir=tmp_path, retries=1)

        with patch.object(fetcher, "fetch", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = (b"data", "json")

            fetcher.fetch_sync(
                "https://example.com/data.json",
                progress_desc="Downloading test",
            )

        mock_fetch.assert_called_once_with(
            "https://example.com/data.json",
            None,
            decompress=True,
            progress_desc="Downloading test",
            headers=None,
        )


async def _async_iter(items: list[bytes]) -> None:  # type: ignore[return]
    """Helper: create an async iterator from a list of items."""
    for item in items:
        yield item  # type: ignore[misc]
