"""HTTP Utilities - Async HTTP client with caching, retry, and format detection.

Provides:
- HttpError: Exception wrapper (prevents httpx from leaking to enrichers/extractors)
- is_url: Check if source is HTTP/HTTPS URL
- detect_format: Detect content format from URL, headers, or content
- HttpFetcher: Async HTTP client with disk caching, gzip support, and retry

Layer: Utility
Dependencies: httpx (internal only, wrapped by HttpError)
"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import logging
import time
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

ContentFormat = Literal["json", "jsonl", "csv", "unknown"]


class HttpError(Exception):
    """HTTP operation failed.

    Wraps httpx exceptions to prevent httpx from leaking into enrichers/extractors.
    Enrichers and extractors should import and catch this instead of httpx.HTTPError.

    Examples:
        >>> from vulnflow.utils.http import HttpError, HttpFetcher
        >>>
        >>> fetcher = HttpFetcher()
        >>> try:
        ...     content, fmt = await fetcher.fetch("https://example.com/data.json")
        ... except HttpError as e:
        ...     logger.warning("Failed to fetch data: %s", e)
    """

    pass


def is_url(source: str) -> bool:
    """Check if source is an HTTP/HTTPS URL.

    Args:
        source: Path or URL string to check

    Returns:
        True if source starts with http:// or https://
    """
    return source.startswith(("http://", "https://"))


def detect_format(
    url: str, content_type: str | None = None, content: bytes | None = None
) -> ContentFormat:
    """Detect content format from URL extension, Content-Type, or content sniffing.

    Detection order:
    1. URL extension (.json, .jsonl, .ndjson, .csv)
    2. Content-Type header
    3. Content sniffing (first non-whitespace bytes)

    Args:
        url: URL string (used for extension detection)
        content_type: HTTP Content-Type header value
        content: Raw content bytes for sniffing

    Returns:
        Detected format: "json", "jsonl", "csv", or "unknown"
    """
    # Strip .gz suffix for extension detection
    url_lower = url.lower()
    if url_lower.endswith(".gz"):
        url_lower = url_lower[:-3]

    # 1. Check URL extension
    if url_lower.endswith(".json"):
        return "json"
    if url_lower.endswith((".jsonl", ".ndjson")):
        return "jsonl"
    if url_lower.endswith(".csv"):
        return "csv"

    # 2. Check Content-Type header
    if content_type:
        ct_lower = content_type.lower()
        if "json" in ct_lower:
            # application/json, application/x-ndjson, etc.
            if "ndjson" in ct_lower or "jsonl" in ct_lower:
                return "jsonl"
            return "json"
        if "csv" in ct_lower or "text/csv" in ct_lower:
            return "csv"

    # 3. Content sniffing
    if content:
        # Skip BOM and whitespace
        stripped = content.lstrip(b"\xef\xbb\xbf \t\n\r")
        if stripped:
            first_char = chr(stripped[0])
            # JSON starts with { or [
            if first_char in "{[":
                return "json"
            # CSV typically starts with letters or quotes
            if first_char.isalpha() or first_char == '"':
                return "csv"

    return "unknown"


class HttpFetcher:
    """Async HTTP client with disk caching, retry, and gzip support.

    Features:
    - Automatic gzip decompression (.gz extension or magic bytes)
    - Disk-based caching with TTL
    - Hash-based cache filenames for URL safety
    - Retry with exponential backoff for transient failures
    """

    GZIP_MAGIC = b"\x1f\x8b"

    def __init__(
        self,
        cache_dir: Path | str = ".cache",
        timeout: float = 120.0,
        retries: int = 3,
        backoff_base: float = 1.0,
    ) -> None:
        """Initialize HTTP fetcher.

        Args:
            cache_dir: Directory for cached responses
            timeout: HTTP request timeout in seconds
            retries: Maximum number of attempts for transient failures
            backoff_base: Base delay in seconds for exponential backoff
        """
        self.cache_dir = Path(cache_dir)
        self.timeout = timeout
        self._retries = retries
        self._backoff_base = backoff_base

    async def fetch(
        self, url: str, cache_ttl: int | None = None
    ) -> tuple[bytes, ContentFormat]:
        """Fetch URL with optional caching, retry, and auto-decompress gzip.

        Retries on transient failures (connection errors, timeouts, 5xx
        responses) with exponential backoff. Client errors (4xx) raise
        immediately.

        Args:
            url: HTTP/HTTPS URL to fetch
            cache_ttl: Cache time-to-live in seconds. None disables caching.

        Returns:
            Tuple of (decompressed content bytes, detected format)

        Raises:
            HttpError: On network or HTTP errors (wraps httpx exceptions)
        """
        import httpx

        # Check cache first (no retry needed for local I/O)
        if cache_ttl is not None:
            cached = self._read_cache(url, cache_ttl)
            if cached is not None:
                content_type = None  # No header for cached content
                fmt = detect_format(url, content_type, cached)
                return cached, fmt

        # Fetch from network with retry
        last_exc: Exception | None = None
        for attempt in range(self._retries):
            try:
                async with httpx.AsyncClient(
                    timeout=self.timeout, follow_redirects=True
                ) as client:
                    response = await client.get(url)
                    response.raise_for_status()
                break  # Success
            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                last_exc = exc
                if attempt < self._retries - 1:
                    wait = self._backoff_base * (2 ** attempt)
                    logger.warning(
                        "HTTP fetch attempt %d/%d failed for %s: %s. "
                        "Retrying in %.1fs",
                        attempt + 1,
                        self._retries,
                        url,
                        exc,
                        wait,
                    )
                    await asyncio.sleep(wait)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code >= 500 and attempt < self._retries - 1:
                    last_exc = exc
                    wait = self._backoff_base * (2 ** attempt)
                    logger.warning(
                        "HTTP fetch attempt %d/%d got %d for %s. "
                        "Retrying in %.1fs",
                        attempt + 1,
                        self._retries,
                        exc.response.status_code,
                        url,
                        wait,
                    )
                    await asyncio.sleep(wait)
                else:
                    # 4xx errors or last attempt for 5xx — raise immediately
                    raise HttpError(
                        f"Failed to fetch {url}: {exc}"
                    ) from exc
        else:
            # All retries exhausted
            raise HttpError(f"Failed to fetch {url}: {last_exc}") from last_exc

        content = response.content
        content_type = response.headers.get("content-type")

        # Auto-decompress gzip
        content = self._maybe_decompress(content, url)

        # Cache if TTL specified
        if cache_ttl is not None:
            self._write_cache(url, content)

        fmt = detect_format(url, content_type, content)
        return content, fmt

    def fetch_sync(
        self, url: str, cache_ttl: int | None = None
    ) -> tuple[bytes, ContentFormat]:
        """Synchronous wrapper for fetch().

        Handles Jupyter/IPython environments where an event loop is already
        running by using nest_asyncio to allow nested asyncio.run() calls.

        Args:
            url: HTTP/HTTPS URL to fetch
            cache_ttl: Cache time-to-live in seconds. None disables caching.

        Returns:
            Tuple of (decompressed content bytes, detected format)

        Raises:
            HttpError: On network or HTTP errors (wraps httpx exceptions)
        """
        try:
            asyncio.get_running_loop()
            # Already in an event loop (Jupyter) — need nest_asyncio
            import nest_asyncio

            nest_asyncio.apply()
        except RuntimeError:
            pass  # No loop running, asyncio.run() will work fine
        return asyncio.run(self.fetch(url, cache_ttl))

    def _maybe_decompress(self, content: bytes, url: str) -> bytes:
        """Decompress gzip content if detected.

        Detection:
        1. .gz URL extension
        2. Gzip magic bytes (0x1f 0x8b)

        Args:
            content: Raw content bytes
            url: Source URL for extension check

        Returns:
            Decompressed content if gzip, otherwise original content
        """
        is_gzip_url = url.lower().endswith(".gz")
        is_gzip_magic = content[:2] == self.GZIP_MAGIC

        if is_gzip_url or is_gzip_magic:
            try:
                return gzip.decompress(content)
            except gzip.BadGzipFile:
                logger.warning("Failed to decompress gzip content from %s", url)
                return content

        return content

    def _cache_path(self, url: str) -> Path:
        """Generate cache file path for URL.

        Uses SHA256 hash to create safe filenames.

        Args:
            url: URL to hash

        Returns:
            Path to cache file
        """
        url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
        return self.cache_dir / f"{url_hash}.cache"

    def _read_cache(self, url: str, ttl: int) -> bytes | None:
        """Read cached content if fresh.

        Args:
            url: URL used as cache key
            ttl: Maximum age in seconds

        Returns:
            Cached content if fresh, None otherwise
        """
        cache_path = self._cache_path(url)
        if not cache_path.exists():
            return None

        # Check age
        age = time.time() - cache_path.stat().st_mtime
        if age > ttl:
            return None

        return cache_path.read_bytes()

    def _write_cache(self, url: str, content: bytes) -> None:
        """Write content to cache.

        Args:
            url: URL used as cache key
            content: Content bytes to cache
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self._cache_path(url)
        cache_path.write_bytes(content)
