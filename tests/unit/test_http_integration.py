"""Integration tests for HTTP download functionality in extractors/enrichers."""

from __future__ import annotations

import gzip
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from vulnflow.extractors.nvd import NVDExtractor
from vulnflow.transforms.enrichers.epss import DEFAULT_EPSS_URL, EPSSEnricher
from vulnflow.transforms.enrichers.kev import DEFAULT_KEV_URL, KEVEnricher
from vulnflow.utils.http import HttpFetcher


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

    def test_nvd_load_from_url_mocked(self, nvd_json_content: bytes) -> None:
        with patch.object(HttpFetcher, "fetch_sync") as mock_fetch:
            mock_fetch.return_value = (nvd_json_content, "json")

            extractor = NVDExtractor("https://example.com/nvd.json")
            content = extractor._load_from_url("https://example.com/nvd.json")

            assert content == nvd_json_content

    def test_nvd_extract_from_url_mocked(self, nvd_json_content: bytes) -> None:
        with patch.object(HttpFetcher, "fetch_sync") as mock_fetch:
            mock_fetch.return_value = (nvd_json_content, "json")

            extractor = NVDExtractor("https://example.com/nvd.json")
            dataset = extractor.extract()

            assert len(dataset) == 1
            cve_ids = dataset.get_column("cve_id")
            assert cve_ids[0] == "CVE-2021-44228"

    def test_nvd_load_gzip_file(self, tmp_path: Path, nvd_json_content: bytes) -> None:
        # Create gzipped file
        gz_path = tmp_path / "nvd.json.gz"
        gz_path.write_bytes(gzip.compress(nvd_json_content))

        extractor = NVDExtractor(str(gz_path))
        content = extractor._load_from_file(gz_path)

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
