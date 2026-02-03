"""Tests for HTTP utilities - is_url, detect_format, HttpFetcher."""

from __future__ import annotations

import gzip
import time
from pathlib import Path

from vulnflow.utils.http import HttpFetcher, detect_format, is_url


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
